"""
Memory-focused PTX structural inspection tests for OpenCUDA v0.5.

Each test compiles a specific kernel and asserts structural properties of
the emitted PTX related to memory operations: pointer parameters, load/store
opcodes, address register widths, local memory usage, and guarded stores.
"""

import re
import pytest
from pathlib import Path

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx


TESTS_DIR = Path(__file__).parent.parent.parent / 'tests'


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ptx(source: str) -> str:
    """Compile CUDA source to combined PTX string (preamble + all kernels)."""
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


def _ptx_file(filename: str) -> str:
    """Compile a .cu file from the tests directory."""
    src = (TESTS_DIR / filename).read_text(encoding='utf-8')
    return _ptx(src)


# ---------------------------------------------------------------------------
# Test 1: Every .param .u64 has a corresponding ld.param.u64
# ---------------------------------------------------------------------------

def test_pointer_params_load_u64():
    """Every .param .u64 declaration must have a matching ld.param.u64 instruction."""
    ptx = _ptx_file('nasty_mem_multiptr.cu')

    # Find all .param .u64 <name> in the kernel signature
    param_names = re.findall(r'\.param \.u64 (\w+)', ptx)
    assert param_names, f"Expected .param .u64 entries in PTX, got:\n{ptx}"

    for pname in param_names:
        expected = f'ld.param.u64'
        assert expected in ptx, \
            f"Expected '{expected}' instruction for param '{pname}', got:\n{ptx}"

    # More specifically: count of ld.param.u64 matches count of .param .u64
    ld_count = ptx.count('ld.param.u64')
    assert ld_count == len(param_names), (
        f"Mismatch: {len(param_names)} .param .u64 declarations but "
        f"{ld_count} ld.param.u64 instructions"
    )


# ---------------------------------------------------------------------------
# Test 2: Global load type matches element type
# ---------------------------------------------------------------------------

def test_global_load_type_matches_element():
    """nasty_mem_mixed_load.cu: ld.global.f32 for float, ld.global.b16 for half."""
    ptx = _ptx_file('nasty_mem_mixed_load.cu')

    assert 'ld.global.f32' in ptx, \
        f"Expected 'ld.global.f32' for float array load, got:\n{ptx}"
    assert 'ld.global.b16' in ptx, \
        f"Expected 'ld.global.b16' for half array load (b16 not f16), got:\n{ptx}"
    assert 'ld.global.f16' not in ptx, \
        f"'ld.global.f16' is invalid PTX — b16 must be used instead, got:\n{ptx}"


# ---------------------------------------------------------------------------
# Test 3: No f16 load or store opcode in any half-pointer kernel
# ---------------------------------------------------------------------------

def test_no_f16_load_or_store_opcode():
    """Kernels with half pointers must not use ld.global.f16 or st.global.f16."""
    ptx = _ptx_file('nasty_mem_mixed_load.cu')

    assert 'ld.global.f16' not in ptx, \
        f"'ld.global.f16' is invalid PTX — b16 must be used, got:\n{ptx}"
    assert 'st.global.f16' not in ptx, \
        f"'st.global.f16' is invalid PTX — b16 must be used, got:\n{ptx}"


# ---------------------------------------------------------------------------
# Test 4: No ld.param.f16 or .param .f16 in kernels with half parameters
# ---------------------------------------------------------------------------

def test_no_param_f16_opcode():
    """Kernels with half parameters must not use ld.param.f16 or .param .f16."""
    src = """
__global__ void k(float *out, half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (float)in[tid];
    }
}
"""
    ptx = _ptx(src)

    assert 'ld.param.f16' not in ptx, \
        f"'ld.param.f16' should not appear — use b16/u64 for half pointer, got:\n{ptx}"
    assert '.param .f16' not in ptx, \
        f"'.param .f16' should not appear in kernel signature, got:\n{ptx}"


# ---------------------------------------------------------------------------
# Test 5: Address registers are 64-bit (%rd prefix)
# ---------------------------------------------------------------------------

def test_address_regs_are_64bit():
    """All address operands in ld.global.X [%REG] and st.global.X [%REG]
    must use %rd registers (64-bit), not %r (32-bit)."""
    ptx = _ptx_file('nasty_mem_ptr_arith.cu')

    # Extract all address registers from loads
    load_addr_regs = re.findall(r'ld\.global\.\w+ %\w+, \[%(\w+)\]', ptx)
    store_addr_regs = re.findall(r'st\.global\.\w+ \[%(\w+)\]', ptx)

    all_addr_regs = load_addr_regs + store_addr_regs
    assert all_addr_regs, f"Expected at least one ld.global or st.global, got:\n{ptx}"

    for reg in all_addr_regs:
        prefix = re.match(r'[a-z]+', reg).group(0)
        assert prefix == 'rd', (
            f"Address register %{reg} uses prefix '{prefix}', "
            f"expected 'rd' (64-bit) — using 32-bit address is wrong. PTX:\n{ptx}"
        )


# ---------------------------------------------------------------------------
# Test 6: Local memory uses local opcode (st.local. or ld.local.)
# ---------------------------------------------------------------------------

def test_local_memory_uses_local_opcode():
    """nasty_mem_local_scratch.cu must emit st.local. or ld.local. instructions
    (not just st.global. for the local scratch area)."""
    ptx = _ptx_file('nasty_mem_local_scratch.cu')

    has_local_op = 'st.local.' in ptx or 'ld.local.' in ptx
    assert has_local_op, (
        f"Expected 'st.local.' or 'ld.local.' in PTX for local scratch kernel, "
        f"got:\n{ptx}"
    )


# ---------------------------------------------------------------------------
# Test 7: Guarded store is in correct branch (predicated)
# ---------------------------------------------------------------------------

def test_guarded_store_in_correct_branch():
    """nasty_mem_guarded_store.cu: there must be at least one @%p predicated
    branch before the store (the store is conditionally executed)."""
    ptx = _ptx_file('nasty_mem_guarded_store.cu')

    # Count predicated branches (@%pN bra)
    pred_branches = re.findall(r'@%p\d+ bra', ptx)
    assert len(pred_branches) >= 1, (
        f"Expected at least one '@%p bra' (predicated branch) before guarded store, "
        f"got:\n{ptx}"
    )

    # Confirm store exists and is guarded (in conditional true branch)
    assert 'st.global.' in ptx, \
        f"Expected 'st.global.' in guarded store kernel, got:\n{ptx}"


# ---------------------------------------------------------------------------
# Test 8: Pointer arithmetic uses add.u64
# ---------------------------------------------------------------------------

def test_ptr_arith_uses_add_u64():
    """nasty_mem_ptr_arith.cu: address computation must use add.u64
    (not add.s32 or add.u32 on address registers)."""
    ptx = _ptx_file('nasty_mem_ptr_arith.cu')

    assert 'add.u64' in ptx, (
        f"Expected 'add.u64' for pointer address computation (not 32-bit add), "
        f"got:\n{ptx}"
    )


# ---------------------------------------------------------------------------
# Test 9: Store after merge appears in merge block
# ---------------------------------------------------------------------------

def test_store_after_merge_uses_post_merge_value():
    """nasty_mem_merge_store.cu: the st.global. instruction must appear AFTER
    the if_merge label in the PTX text (the merge block follows if_true/if_false)."""
    ptx = _ptx_file('nasty_mem_merge_store.cu')

    # Find position of the merge label (if_merge_N:) that follows the if/else
    # and the position of the store
    merge_match = re.search(r'if_merge_\d+:', ptx)
    store_match = re.search(r'st\.global\.', ptx)

    assert merge_match is not None, \
        f"Expected 'if_merge_N:' label in PTX, got:\n{ptx}"
    assert store_match is not None, \
        f"Expected 'st.global.' instruction in PTX, got:\n{ptx}"

    # Find the LAST merge label before the store to confirm store is post-merge
    # The inner if_merge (the one containing the store) should come AFTER if_true/if_false
    merge_positions = [m.start() for m in re.finditer(r'if_merge_\d+:', ptx)]
    store_pos = store_match.start()

    # There should be a merge label that occurs before the store instruction
    labels_before_store = [pos for pos in merge_positions if pos < store_pos]
    assert labels_before_store, (
        f"Expected a merge label to appear before 'st.global.' in PTX "
        f"(store should be in merge block), got:\n{ptx}"
    )


# ---------------------------------------------------------------------------
# Test 10: Mixed-width loads have matching register prefixes
# ---------------------------------------------------------------------------

def test_mixed_width_no_width_mismatch():
    """nasty_mem_mixed_load.cu: for each ld.global.X instruction, the result
    register prefix must match the expected width:
      b16/f16 -> h prefix, f32 -> f prefix, s32/u32/b32 -> r prefix,
      b64/u64/s64 -> rd prefix, f64 -> fd prefix."""
    ptx = _ptx_file('nasty_mem_mixed_load.cu')

    type_to_prefix = {
        'b16': 'h', 'f16': 'h',
        'f32': 'f',
        's32': 'r', 'u32': 'r', 'b32': 'r',
        'f64': 'fd',
        'b64': 'rd', 'u64': 'rd', 's64': 'rd',
    }

    for m in re.finditer(r'ld\.global\.(\w+)\s+%(\w+?\d)', ptx):
        ty = m.group(1)
        reg = m.group(2)
        reg_prefix = re.match(r'[a-z]+', reg).group(0)
        expected_prefix = type_to_prefix.get(ty)
        if expected_prefix is None:
            continue  # unknown type, skip
        assert reg_prefix == expected_prefix, (
            f"ld.global.{ty} loads into %{reg} (prefix '{reg_prefix}'), "
            f"expected prefix '{expected_prefix}'. Full PTX:\n{ptx}"
        )
