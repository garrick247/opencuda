"""
Memory invariant tests for OpenCUDA v0.5.

Checks memory operation correctness properties across ALL kernel files.
These invariants must hold for every .cu file in tests/ (excluding gpu_ prefixed).
"""

import re
import pytest
from pathlib import Path

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx


TESTS_DIR = Path(__file__).parent.parent.parent / 'tests'
ALL_KERNELS = [f for f in sorted(TESTS_DIR.glob('*.cu')) if not f.name.startswith('gpu_')]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ptx(source: str) -> str:
    """Compile CUDA source to combined PTX string."""
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
    """Return {prefix: max_declared_count} from .reg declarations."""
    decls = {}
    for m in re.finditer(r'\.reg \.\w+ %(\w+)<(\d+)>', ptx):
        key, count = m.group(1), int(m.group(2))
        decls[key] = max(decls.get(key, 0), count)
    return decls


# ---------------------------------------------------------------------------
# Invariant 1: No f16 opcode in param or store
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_KERNELS, ids=[f.stem for f in ALL_KERNELS])
def test_no_f16_opcode_in_param_or_store(cu_file):
    """No kernel should emit ld.param.f16, st.global.f16, or .param .f16 in PTX.
    Half values must use b16 for loads/stores and u64 for pointer params."""
    source = cu_file.read_text(encoding='utf-8')
    ptx = _ptx(source)

    assert 'ld.param.f16' not in ptx, (
        f"{cu_file.name}: 'ld.param.f16' found — use b16 or u64 for half params"
    )
    assert 'st.global.f16' not in ptx, (
        f"{cu_file.name}: 'st.global.f16' found — use st.global.b16 instead"
    )
    assert '.param .f16' not in ptx, (
        f"{cu_file.name}: '.param .f16' found in kernel signature — invalid PTX"
    )


# ---------------------------------------------------------------------------
# Invariant 2: Global store address registers are 64-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_KERNELS, ids=[f.stem for f in ALL_KERNELS])
def test_global_store_addr_is_64bit(cu_file):
    """All address operands in st.global.X [%REG] must use %rd (64-bit) registers."""
    source = cu_file.read_text(encoding='utf-8')
    ptx = _ptx(source)

    store_addr_regs = re.findall(r'st\.global\.\w+ \[%(\w+)\]', ptx)
    if not store_addr_regs:
        pytest.skip(f"{cu_file.name}: no global stores, nothing to check")

    for reg in store_addr_regs:
        prefix = re.match(r'[a-z]+', reg).group(0)
        assert prefix == 'rd', (
            f"{cu_file.name}: st.global address register %{reg} uses prefix "
            f"'{prefix}' — expected 'rd' (64-bit pointer)"
        )


# ---------------------------------------------------------------------------
# Invariant 3: Global load address registers are 64-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_KERNELS, ids=[f.stem for f in ALL_KERNELS])
def test_global_load_addr_is_64bit(cu_file):
    """All address operands in ld.global.X [%REG] must use %rd (64-bit) registers."""
    source = cu_file.read_text(encoding='utf-8')
    ptx = _ptx(source)

    load_addr_regs = re.findall(r'ld\.global[\.\w]+ %\w+, \[%(\w+)\]', ptx)
    if not load_addr_regs:
        pytest.skip(f"{cu_file.name}: no global loads, nothing to check")

    for reg in load_addr_regs:
        prefix = re.match(r'[a-z]+', reg).group(0)
        assert prefix == 'rd', (
            f"{cu_file.name}: ld.global address register %{reg} uses prefix "
            f"'{prefix}' — expected 'rd' (64-bit pointer)"
        )


# ---------------------------------------------------------------------------
# Invariant 4: No mixed-width float store (half reg via f32 opcode or vice versa)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_KERNELS, ids=[f.stem for f in ALL_KERNELS])
def test_no_mixed_width_float_store(cu_file):
    """Assert no st.global.f32 with %h (half) register, and no st.global.f16
    with %f (float) register — opcode width must match register width."""
    source = cu_file.read_text(encoding='utf-8')
    ptx = _ptx(source)

    # st.global.f32 [...], %hN  -- storing half via f32 opcode
    bad_f32_half = re.findall(r'st\.global\.f32 \[%\w+\], %h\d+', ptx)
    assert not bad_f32_half, (
        f"{cu_file.name}: found st.global.f32 storing half register: {bad_f32_half}"
    )

    # st.global.f16 [...], %fN  -- storing float via f16 opcode
    bad_f16_float = re.findall(r'st\.global\.f16 \[%\w+\], %f\d+', ptx)
    assert not bad_f16_float, (
        f"{cu_file.name}: found st.global.f16 storing float register: {bad_f16_float}"
    )


# ---------------------------------------------------------------------------
# Invariant 5: Pointer parameter declarations have matching ld.param.u64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_KERNELS, ids=[f.stem for f in ALL_KERNELS])
def test_param_pointer_decl_uses_u64(cu_file):
    """For each .param .u64 (pointer param) in a kernel signature, there must
    be a corresponding ld.param.u64 instruction that loads it."""
    source = cu_file.read_text(encoding='utf-8')
    ptx = _ptx(source)

    # Exclude forward-declaration lines (single-line .func signatures ending in ';').
    # Those have .param .u64 in the signature but no body and no ld.param instructions.
    non_fwd_ptx = '\n'.join(
        line for line in ptx.splitlines()
        if not line.rstrip().endswith(';')
    )
    param_u64_count = len(re.findall(r'\.param \.u64 \w+', non_fwd_ptx))
    if param_u64_count == 0:
        pytest.skip(f"{cu_file.name}: no .param .u64 declarations, nothing to check")

    ld_u64_count = ptx.count('ld.param.u64')

    assert ld_u64_count == param_u64_count, (
        f"{cu_file.name}: {param_u64_count} .param .u64 declarations but "
        f"{ld_u64_count} ld.param.u64 instructions — mismatch"
    )
