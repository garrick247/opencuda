#!/usr/bin/env python3
"""
benchmark_regalloc.py — Register allocation quality across real kernels.

Measures for each kernel:
  - Declared register count per type prefix (r/rd/f/fd/p/h)
  - Total instruction count in emitted PTX
  - Naive SSA register count (what allocation without linear scan would use)
  - Reduction factor: naive / declared (higher = more effective reuse)
  - Gap ratio: declared / distinctly used (lower = tighter packing)

Run from repo root:
    python tools/benchmark_regalloc.py
    python tools/benchmark_regalloc.py --sort reduction
    python tools/benchmark_regalloc.py --min-instructions 10
    python tools/benchmark_regalloc.py --csv > bench.csv
"""

import re
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = REPO_ROOT / 'tests'
sys.path.insert(0, str(REPO_ROOT))

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx


# ---------------------------------------------------------------------------
# PTX analysis helpers
# ---------------------------------------------------------------------------

def extract_reg_decls(ptx: str) -> dict[str, int]:
    """Per-kernel max declared count for each register prefix."""
    # Split into per-kernel sections so we take the MAX across all kernels
    # (multi-kernel files have independent .reg declarations per entry)
    decls: dict[str, int] = {}
    for m in re.finditer(r'\.reg \.\w+ %([a-z]+)<(\d+)>', ptx):
        key, count = m.group(1), int(m.group(2))
        decls[key] = max(decls.get(key, 0), count)
    return decls


def extract_reg_refs(ptx: str) -> dict[str, set[int]]:
    """Distinct register indices used per prefix (across all kernels)."""
    refs: dict[str, set[int]] = {}
    for m in re.finditer(r'%([a-z]+)(\d+)', ptx):
        refs.setdefault(m.group(1), set()).add(int(m.group(2)))
    return refs


def count_instructions(ptx: str) -> int:
    """Count PTX instruction lines (indented non-empty lines that aren't .reg/.param/labels)."""
    count = 0
    for line in ptx.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('.') or stripped.startswith('//'):
            continue
        if stripped.endswith(':'):
            continue
        if stripped.startswith('{') or stripped.startswith('}'):
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Per-file benchmark
# ---------------------------------------------------------------------------

def benchmark_file(cu_path: Path) -> list[dict]:
    """Returns a list of row-dicts, one per kernel in the file."""
    source = cu_path.read_text(encoding='utf-8')
    try:
        source = preprocess(source)
        module = parse(source)
        # Record naive SSA id counts BEFORE optimization (worst case)
        naive_pre = {k.name: k._next_id for k in module.kernels}
        module = optimize(module)
        naive_post = {k.name: k._next_id for k in module.kernels}
        ptx_map = ir_to_ptx(module)
    except Exception as exc:
        return [{'file': cu_path.name, 'kernel': '(error)', 'error': str(exc)}]

    rows = []
    for kernel_name, ptx_text in ptx_map.items():
        if kernel_name.startswith('__'):
            continue

        decls = extract_reg_decls(ptx_text)
        refs = extract_reg_refs(ptx_text)
        n_insts = count_instructions(ptx_text)
        naive = naive_post.get(kernel_name, naive_pre.get(kernel_name, 0))

        # Primary regs for reduction/gap analysis: r (b32) and f (f32)
        total_declared = sum(decls.values())
        total_distinct_used = sum(len(s) for s in refs.values())

        reduction = naive / total_declared if total_declared > 0 else 0.0
        gap_ratio = total_declared / total_distinct_used if total_distinct_used > 0 else 1.0

        row = {
            'file': cu_path.name,
            'kernel': kernel_name,
            'n_insts': n_insts,
            'naive_ssa': naive,
            'total_declared': total_declared,
            'distinct_used': total_distinct_used,
            'reduction': reduction,
            'gap_ratio': gap_ratio,
            'decls': decls,
            'error': None,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_decls(decls: dict[str, int]) -> str:
    """Compact summary: r4/rd3/f2/p1"""
    parts = []
    for prefix in ('r', 'rd', 'f', 'fd', 'h', 'p'):
        if prefix in decls:
            parts.append(f'{prefix}{decls[prefix]}')
    return '/'.join(parts) if parts else '-'


def print_table(rows: list[dict], sort_key: str = 'file') -> None:
    valid = [r for r in rows if r['error'] is None]
    errors = [r for r in rows if r['error'] is not None]

    if sort_key == 'reduction':
        valid.sort(key=lambda r: -r['reduction'])
    elif sort_key == 'instructions':
        valid.sort(key=lambda r: -r['n_insts'])
    elif sort_key == 'declared':
        valid.sort(key=lambda r: -r['total_declared'])
    else:
        valid.sort(key=lambda r: (r['file'], r['kernel']))

    # Header
    w_file = max(30, max((len(r['file']) for r in valid), default=10))
    w_kern = max(18, max((len(r['kernel']) for r in valid), default=10))
    hdr = (
        f"{'File':<{w_file}}  {'Kernel':<{w_kern}}"
        f"  {'Insts':>6}  {'Naive':>5}  {'Decl':>5}  {'Used':>5}"
        f"  {'Reduc':>5}  {'Gap':>5}  Regs"
    )
    print(hdr)
    print('-' * len(hdr))

    for r in valid:
        print(
            f"{r['file']:<{w_file}}  {r['kernel']:<{w_kern}}"
            f"  {r['n_insts']:>6}  {r['naive_ssa']:>5}  {r['total_declared']:>5}"
            f"  {r['distinct_used']:>5}  {r['reduction']:>5.2f}x  {r['gap_ratio']:>4.2f}"
            f"  {format_decls(r['decls'])}"
        )

    if errors:
        print(f"\n{len(errors)} file(s) with errors:")
        for r in errors:
            print(f"  {r['file']}: {r['error']}")

    # Summary stats
    if valid:
        avg_reduction = sum(r['reduction'] for r in valid) / len(valid)
        avg_gap = sum(r['gap_ratio'] for r in valid) / len(valid)
        max_insts = max(r['n_insts'] for r in valid)
        total_kernels = len(valid)
        print(f"\nSummary: {total_kernels} kernels | "
              f"avg reduction {avg_reduction:.2f}x | "
              f"avg gap {avg_gap:.2f} | "
              f"max instructions {max_insts}")


def print_csv(rows: list[dict]) -> None:
    print("file,kernel,n_insts,naive_ssa,total_declared,distinct_used,"
          "reduction,gap_ratio,r,rd,f,fd,h,p")
    for r in rows:
        if r['error']:
            continue
        d = r['decls']
        print(
            f"{r['file']},{r['kernel']},{r['n_insts']},{r['naive_ssa']},"
            f"{r['total_declared']},{r['distinct_used']},"
            f"{r['reduction']:.4f},{r['gap_ratio']:.4f},"
            f"{d.get('r',0)},{d.get('rd',0)},{d.get('f',0)},"
            f"{d.get('fd',0)},{d.get('h',0)},{d.get('p',0)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sort', choices=['file', 'reduction', 'instructions', 'declared'],
                        default='file', help='Sort column (default: file)')
    parser.add_argument('--min-instructions', type=int, default=0,
                        help='Skip kernels with fewer than N PTX instructions')
    parser.add_argument('--csv', action='store_true',
                        help='Emit CSV to stdout instead of table')
    parser.add_argument('kernels', nargs='*', metavar='FILE.cu',
                        help='Specific .cu files (default: all non-gpu tests/*.cu)')
    args = parser.parse_args()

    if args.kernels:
        cu_files = [Path(f) for f in args.kernels]
    else:
        cu_files = sorted(
            f for f in TESTS_DIR.glob('*.cu')
            if not f.name.startswith('gpu_')
        )

    all_rows = []
    for cu_file in cu_files:
        all_rows.extend(benchmark_file(cu_file))

    if args.min_instructions > 0:
        all_rows = [
            r for r in all_rows
            if r['error'] or r['n_insts'] >= args.min_instructions
        ]

    if args.csv:
        print_csv(all_rows)
    else:
        print_table(all_rows, sort_key=args.sort)


if __name__ == '__main__':
    main()
