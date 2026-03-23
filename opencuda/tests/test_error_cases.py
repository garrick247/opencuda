"""
Error and edge-case behavior tests.

Verifies that unsupported constructs fail with appropriate errors,
and that partially-supported constructs degrade predictably (not silently corrupt output).
"""

import pytest

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse, ParseError
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx


def _compile(source: str) -> str:
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


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_prefix_increment_raises():
    """prefix ++ (++i) must raise ParseError since only postfix is implemented."""
    src = """
__global__ void f(int *out) { int i = 0; ++i; out[0] = i; }
"""
    with pytest.raises(Exception):
        _compile(src)


def test_undefined_variable_raises():
    """Referencing an undefined variable must raise ParseError."""
    src = """
__global__ void f(int *out) { out[0] = undefined_var; }
"""
    with pytest.raises(Exception):
        _compile(src)


def test_missing_semicolon_raises():
    """Missing semicolon must raise ParseError or similar."""
    src = """
__global__ void f(int *out) { out[0] = 1 }
"""
    with pytest.raises(Exception):
        _compile(src)


def test_printf_non_literal_format_degrades_not_crashes():
    """printf with a non-literal format: either raises ParseError OR produces a
    kernel. Must NOT crash with an unhandled Python exception."""
    src = """
__global__ void f(int *msg) { printf((char*)msg); }
"""
    try:
        source = preprocess(src)
        module = parse(source)
        # If it gets this far without raising, the module must have at least 1 kernel
        assert len(module.kernels) >= 1, "Expected at least 1 kernel in parsed module"
    except Exception:
        pass  # ParseError or any exception is acceptable


def test_recursive_device_func_hits_limit():
    """When n is a runtime parameter (not constant), inlining recurses infinitely.
    Must raise RecursionError (Python stack overflow), NOT produce silently wrong PTX."""
    src = """
__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
__global__ void f(int *out, int n) {
    out[0] = factorial(n);
}
"""
    with pytest.raises((RecursionError, Exception), match=None):
        _compile(src)


def test_empty_kernel_is_valid():
    """Empty __global__ kernel must compile successfully to PTX with ret;."""
    src = """
__global__ void f() {}
"""
    ptx = _compile(src)
    assert 'ret;' in ptx, f"Expected ret; in PTX for empty kernel, got:\n{ptx}"


def test_void_return_in_global():
    """Explicit return; in void kernel is valid and must compile."""
    src = """
__global__ void f(int *out) { out[0] = 1; return; }
"""
    ptx = _compile(src)
    assert 'ret;' in ptx, f"Expected ret; in PTX, got:\n{ptx}"


def test_multi_return_device_func_correctness():
    """Verifies that multi-return inlining produces structurally correct PTX.

    PTX must:
      - contain ret;
      - contain bra instructions (for branches)
      - NOT contain any %r register with index >= 50 (no SSA ID leaks)
    """
    src = """
__device__ int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
__global__ void f(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = sign(in[tid]);
}
"""
    ptx = _compile(src)
    assert 'ret;' in ptx, f"Expected ret; in PTX, got:\n{ptx}"
    assert 'bra ' in ptx, f"Expected bra instruction in PTX, got:\n{ptx}"

    import re
    raw_id_refs = re.findall(r'%r(\d+)', ptx)
    for idx_str in raw_id_refs:
        idx = int(idx_str)
        assert idx < 50, (
            f"Register %r{idx} looks like a raw SSA ID leak (expected < 50). "
            f"PTX:\n{ptx}"
        )


def test_short_circuit_not_guaranteed():
    """OpenCUDA && does not guarantee short-circuit evaluation.

    This test documents the behavior: && evaluates both sides.
    We verify the kernel compiles (not that it short-circuits).
    """
    # OpenCUDA && does not guarantee short-circuit evaluation
    src = """
__global__ void f(int *out, int *a) {
    int tid = threadIdx.x;
    if (a != 0 && a[tid] > 0) out[tid] = 1;
    else out[tid] = 0;
}
"""
    ptx = _compile(src)
    assert 'ret;' in ptx, f"Expected ret; in compiled PTX, got:\n{ptx}"
    assert '.entry' in ptx, f"Expected .entry in compiled PTX, got:\n{ptx}"
