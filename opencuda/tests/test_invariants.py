"""
Compiler invariant tests.

Asserts structural guarantees that must hold across ALL compilation outputs:
  1. Every referenced register index is within the declared count.
  2. Declared count is less than the naive (SSA-ID-based) count for real kernels.
  3. Gap ratio (declared / distinctly used) stays within threshold.
  4. Every IR value in surviving instructions is covered by the linear scan allocator.
"""

import re
import pytest
from pathlib import Path

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx, _build_alloc_map
from opencuda.ir.nodes import (Value, BinInst, CmpInst, LoadInst, StoreInst,
                                CvtInst, CallInst, ParamInst, CondBrTerm)


TESTS_DIR = Path(__file__).parent.parent.parent / 'tests'
CU_FILES = sorted(TESTS_DIR.glob('*.cu'))
KERNEL_FILES = [f for f in CU_FILES if not f.name.startswith('gpu_')]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ptx(source: str) -> str:
    """Compile to combined PTX string."""
    source = preprocess(source)
    module = parse(source)
    module = optimize(module)
    ptx_map = ir_to_ptx(module)
    parts = []
    if '__preamble__' in ptx_map:
        parts.append(ptx_map['__preamble__'])
    for name, text in ptx_map.items():
        if not name.startswith('__'):
            parts.append(text)
    return '\n'.join(parts)


def _extract_reg_refs(ptx: str) -> dict:
    """Return {prefix: {physical_indices_used}} from PTX body."""
    refs = {}
    for m in re.finditer(r'%([a-z]+)(\d+)', ptx):
        refs.setdefault(m.group(1), set()).add(int(m.group(2)))
    return refs


def _extract_reg_decls(ptx: str) -> dict:
    """Return {prefix: max_declared_count} from .reg declarations.

    Takes the max across all kernels in the PTX so that multi-kernel files
    (where each kernel has its own .reg declarations) don't produce false
    positives when earlier declarations are overwritten by later smaller ones.
    """
    decls = {}
    for m in re.finditer(r'\.reg \.\w+ %(\w+)<(\d+)>', ptx):
        key, count = m.group(1), int(m.group(2))
        decls[key] = max(decls.get(key, 0), count)
    return decls


def _compile_full(source: str):
    """Returns (module, ptx_map, naive_ids). Records naive _next_id BEFORE emission."""
    mod = optimize(parse(preprocess(source)))
    naive_ids = {k.name: k._next_id for k in mod.kernels}
    ptx_map = ir_to_ptx(mod)
    return mod, ptx_map, naive_ids


# ---------------------------------------------------------------------------
# Invariant 1: Every referenced register index is within the declared range
# ---------------------------------------------------------------------------

def test_refs_within_decls_vector_add():
    """vector_add: for every prefix, max(refs[prefix]) < decls[prefix]."""
    source = Path(TESTS_DIR / 'vector_add.cu').read_text(encoding='utf-8')
    ptx = _ptx(source)
    refs = _extract_reg_refs(ptx)
    decls = _extract_reg_decls(ptx)
    for prefix, used_indices in refs.items():
        if prefix not in decls:
            continue
        max_used = max(used_indices)
        declared = decls[prefix]
        assert max_used < declared, (
            f"vector_add: register %{prefix}{max_used} used but only "
            f"%{prefix}<{declared}> declared"
        )


@pytest.mark.parametrize('cu_file', KERNEL_FILES, ids=[f.stem for f in KERNEL_FILES])
def test_refs_within_decls_all_kernels(cu_file):
    """For every prefix seen in PTX body, assert max(used_indices) < declared_count."""
    source = cu_file.read_text(encoding='utf-8')
    ptx = _ptx(source)
    refs = _extract_reg_refs(ptx)
    decls = _extract_reg_decls(ptx)
    for prefix, used_indices in refs.items():
        if prefix not in decls:
            continue
        max_used = max(used_indices)
        declared = decls[prefix]
        assert max_used < declared, (
            f"{cu_file.name}: register %{prefix}{max_used} used but only "
            f"%{prefix}<{declared}> declared"
        )


# ---------------------------------------------------------------------------
# Invariant 2: Linear scan reduces register count vs naive
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="OCUDA-SSA: r-prefix is intentionally SSA-faithful "
                          "(no live-range reuse) to preserve value identity "
                          "through OpenPTXas' LDG dest reorder.  Register "
                          "pressure for b32 is not minimized; correctness "
                          "wins.  Other prefixes still tested.")
@pytest.mark.parametrize('cu_file', KERNEL_FILES, ids=[f.stem for f in KERNEL_FILES])
def test_linear_scan_reduces_b32(cu_file):
    pass


@pytest.mark.parametrize('cu_file', KERNEL_FILES, ids=[f.stem for f in KERNEL_FILES])
def test_linear_scan_reduces_f32(cu_file):
    """For kernels where naive_ids >= 15: declared f register count < naive_ids."""
    source = cu_file.read_text(encoding='utf-8')
    mod, ptx_map, naive_ids = _compile_full(source)
    for kernel_name, ptx_text in ptx_map.items():
        if kernel_name.startswith('__'):
            continue
        n_id = naive_ids.get(kernel_name, 0)
        if n_id < 15:
            continue  # Not enough SSA values to matter
        decls = _extract_reg_decls(ptx_text)
        f_count = decls.get('f', 0)
        if f_count == 0:
            continue  # No f32 registers used
        assert f_count < n_id, (
            f"{cu_file.name}/{kernel_name}: f32 reg count {f_count} should be "
            f"< naive SSA count {n_id}"
        )


# ---------------------------------------------------------------------------
# Invariant 3: Gap ratio within threshold
# ---------------------------------------------------------------------------

def test_gap_ratio_vector_add():
    """vector_add: gap_ratio = decls[prefix] / len(refs[prefix]) must be <= 2.0."""
    source = Path(TESTS_DIR / 'vector_add.cu').read_text(encoding='utf-8')
    ptx = _ptx(source)
    refs = _extract_reg_refs(ptx)
    decls = _extract_reg_decls(ptx)
    for prefix in ('r', 'rd', 'f'):
        if prefix not in refs or prefix not in decls:
            continue
        distinct_used = len(refs[prefix])
        declared = decls[prefix]
        gap_ratio = declared / distinct_used
        assert gap_ratio <= 2.0, (
            f"vector_add: gap ratio for %{prefix} is {gap_ratio:.2f} "
            f"(declared={declared}, distinct_used={distinct_used}), expected <= 2.0"
        )


def test_gap_ratio_register_pressure():
    """register_pressure.cu: f gap_ratio <= 2.5."""
    source = Path(TESTS_DIR / 'register_pressure.cu').read_text(encoding='utf-8')
    ptx = _ptx(source)
    refs = _extract_reg_refs(ptx)
    decls = _extract_reg_decls(ptx)
    prefix = 'f'
    if prefix in refs and prefix in decls:
        distinct_used = len(refs[prefix])
        declared = decls[prefix]
        gap_ratio = declared / distinct_used
        assert gap_ratio <= 2.5, (
            f"register_pressure: gap ratio for %{prefix} is {gap_ratio:.2f} "
            f"(declared={declared}, distinct_used={distinct_used}), expected <= 2.5"
        )


# ---------------------------------------------------------------------------
# Invariant 4: Alloc map covers all IR value refs
# ---------------------------------------------------------------------------

def _check_alloc_covers_kernel(kernel):
    """Assert every Value used in kernel instructions is in the alloc map."""
    alloc, val_type, pred_ids, alloc_max = _build_alloc_map(kernel)

    ir_value_ids = set()
    for bb in kernel.blocks:
        for inst in bb.instructions:
            for attr in ('dest', 'lhs', 'rhs', 'src', 'addr', 'value'):
                v = getattr(inst, attr, None)
                if isinstance(v, Value):
                    ir_value_ids.add(v.id)
            if hasattr(inst, 'args'):
                for a in inst.args:
                    if isinstance(a, Value):
                        ir_value_ids.add(a.id)
        if isinstance(bb.terminator, CondBrTerm):
            cond = bb.terminator.cond
            if isinstance(cond, Value):
                ir_value_ids.add(cond.id)

    unallocated = ir_value_ids - set(alloc.keys())
    assert not unallocated, (
        f"Kernel {kernel.name}: {len(unallocated)} IR values not in alloc map: "
        f"{sorted(unallocated)[:5]}"
    )


def test_alloc_map_covers_ir_values():
    """Every Value appearing in a surviving IR instruction should be
    covered by the linear scan allocator (not fall to the fallback)."""
    source = Path(TESTS_DIR / 'matmul_tiled.cu').read_text(encoding='utf-8')
    mod = optimize(parse(preprocess(source)))
    for kernel in mod.kernels:
        _check_alloc_covers_kernel(kernel)


def test_alloc_map_covers_vector_add():
    """Same as test_alloc_map_covers_ir_values but for vector_add.cu."""
    source = Path(TESTS_DIR / 'vector_add.cu').read_text(encoding='utf-8')
    mod = optimize(parse(preprocess(source)))
    for kernel in mod.kernels:
        _check_alloc_covers_kernel(kernel)


def test_alloc_map_covers_reduce():
    """Same as test_alloc_map_covers_ir_values but for reduce.cu."""
    source = Path(TESTS_DIR / 'reduce.cu').read_text(encoding='utf-8')
    mod = optimize(parse(preprocess(source)))
    for kernel in mod.kernels:
        _check_alloc_covers_kernel(kernel)


# ---------------------------------------------------------------------------
# Invariant 5: Widen cache uses fallback allocator, not raw IDs
# ---------------------------------------------------------------------------

def test_widen_cache_not_raw_id():
    """The widen-cache cvt.u64.u32 should produce a register index
    in the compact range, not equal to the raw SSA value ID."""
    source = Path(TESTS_DIR / 'vector_add.cu').read_text(encoding='utf-8')
    ptx = _ptx(source)
    mod = optimize(parse(preprocess(source)))
    naive = mod.kernels[0]._next_id  # after optimization, before emission
    ptx_map = ir_to_ptx(optimize(parse(preprocess(source))))
    decls = _extract_reg_decls(list(ptx_map.values())[0])
    rd_decl = decls.get('rd', 0)
    # Naive allocation would need rd<naive>, linear scan should need rd<8
    assert rd_decl <= 8, (
        f"Expected <=8 rd registers, got {rd_decl} (naive would be {naive})"
    )
