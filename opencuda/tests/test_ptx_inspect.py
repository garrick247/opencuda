"""
Structural PTX inspection tests for OpenCUDA.

These tests verify specific PTX output properties: register declarations,
instruction selection, printf lowering, __ldg expansion, multi-return
inlining, and register allocation compactness.
"""

import re
import pytest

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx


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


# ---------------------------------------------------------------------------
# f16 / half tests
# ---------------------------------------------------------------------------

def test_half_reg_declaration():
    """half variable should cause .reg .f16 declaration."""
    src = """
__global__ void k(float *out, half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        half v = in[tid];
        out[tid] = (float)v;
    }
}
"""
    ptx = _ptx(src)
    assert '.reg .f16 %h' in ptx, \
        f"Expected '.reg .f16 %h' in PTX, got:\n{ptx}"


def test_half_load_uses_b16():
    """Half loads must use ld.global.b16 (PTX disallows ld.f16)."""
    src = """
__global__ void k(float *out, half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        half v = in[tid];
        out[tid] = (float)v;
    }
}
"""
    ptx = _ptx(src)
    assert 'ld.global.b16' in ptx, \
        f"Expected 'ld.global.b16' in PTX, got:\n{ptx}"
    assert 'ld.global.f16' not in ptx, \
        f"ld.global.f16 is invalid PTX and should not appear"


def test_half_cvt_to_float():
    """(float)half_var should emit cvt.f32.f16."""
    src = """
__global__ void k(float *out, half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        half v = in[tid];
        out[tid] = (float)v;
    }
}
"""
    ptx = _ptx(src)
    assert 'cvt.f32.f16' in ptx, \
        f"Expected 'cvt.f32.f16' instruction in PTX, got:\n{ptx}"


# ---------------------------------------------------------------------------
# printf / vprintf tests
# ---------------------------------------------------------------------------

def test_printf_vprintf_extern():
    """printf should emit an extern vprintf declaration."""
    src = """
__global__ void k(int n) {
    if (threadIdx.x == 0) {
        printf("hello %d\\n", n);
    }
}
"""
    ptx = _ptx(src)
    assert 'vprintf' in ptx, \
        f"Expected vprintf in PTX, got:\n{ptx}"
    assert '.extern' in ptx, \
        f"Expected .extern declaration for vprintf, got:\n{ptx}"


def test_printf_global_format_string():
    """printf format string should appear as a .global .b8 array."""
    src = """
__global__ void k(int n) {
    if (threadIdx.x == 0) {
        printf("hello %d\\n", n);
    }
}
"""
    ptx = _ptx(src)
    assert '.global .align 1 .b8' in ptx, \
        f"Expected '.global .align 1 .b8' format string in PTX, got:\n{ptx}"


def test_printf_call_uni_layout():
    """printf vprintf call should use call.uni and .param .b64."""
    src = """
__global__ void k(int n) {
    if (threadIdx.x == 0) {
        printf("hello %d\\n", n);
    }
}
"""
    ptx = _ptx(src)
    assert 'call.uni' in ptx, \
        f"Expected 'call.uni' in PTX, got:\n{ptx}"
    assert '.param .b64' in ptx, \
        f"Expected '.param .b64' in PTX, got:\n{ptx}"


def test_printf_local_valist_with_args():
    """printf with args should allocate a .local valist buffer."""
    src = """
__global__ void k(int n) {
    if (threadIdx.x == 0) {
        printf("hello %d\\n", n);
    }
}
"""
    ptx = _ptx(src)
    assert '.local .align 8 .b8' in ptx, \
        f"Expected '.local .align 8 .b8' valist in PTX, got:\n{ptx}"


def test_printf_no_valist_for_no_args():
    """printf with no args (just format string) should not allocate a valist."""
    src = """
__global__ void k() {
    if (threadIdx.x == 0) {
        printf("hello\\n");
    }
}
"""
    ptx = _ptx(src)
    assert '.local .align 8 .b8' not in ptx, \
        f"Did not expect '.local .align 8 .b8' for zero-arg printf, got:\n{ptx}"


# ---------------------------------------------------------------------------
# __ldg tests
# ---------------------------------------------------------------------------

def test_ldg_emits_nc_load():
    """__ldg(&ptr[i]) should emit ld.global.nc (non-coherent cache)."""
    src = """
__global__ void k(float *out, const float * __restrict__ in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]);
    }
}
"""
    ptx = _ptx(src)
    assert 'ld.global.nc' in ptx, \
        f"Expected 'ld.global.nc' for __ldg in PTX, got:\n{ptx}"


def test_restrict_stripped_from_ptx():
    """__restrict__ qualifier should not appear in PTX output."""
    src = """
__global__ void k(float *out, const float * __restrict__ in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]);
    }
}
"""
    ptx = _ptx(src)
    assert '__restrict__' not in ptx, \
        f"__restrict__ should not appear in PTX output, got:\n{ptx}"


# ---------------------------------------------------------------------------
# Multi-return inlining tests
# ---------------------------------------------------------------------------

def test_multi_return_two_paths():
    """Device func with 2 return points should produce valid PTX with ret;."""
    src = """
__device__ float clamp_val(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

__global__ void k(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = clamp_val(in[tid]);
    }
}
"""
    ptx = _ptx(src)
    assert '.entry' in ptx, "Expected .entry in PTX"
    assert 'ret;' in ptx, "Expected ret; in PTX"


def test_multi_return_four_paths():
    """Device func with 4 return points (categorize) should produce valid PTX."""
    src = """
__device__ float categorize(float x) {
    if (x < -1.0f) return -2.0f;
    if (x < 0.0f)  return -1.0f;
    if (x < 1.0f)  return x;
    return 2.0f;
}

__global__ void k(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = categorize(in[tid]);
    }
}
"""
    ptx = _ptx(src)
    assert '.entry' in ptx, "Expected .entry in PTX"
    assert 'ret;' in ptx, "Expected ret; in PTX"
    # Should have multiple branches from the inline expansion
    bra_count = ptx.count('bra ')
    assert bra_count >= 4, \
        f"Expected >= 4 branch instructions for 4-way categorize, got {bra_count}"


# ---------------------------------------------------------------------------
# Register allocation compactness tests
# ---------------------------------------------------------------------------

def _count_regs(ptx: str, reg_type: str) -> int:
    """Return declared register count for the given PTX type (e.g. 'b32', 'b64', 'f32')."""
    # Match: .reg .b32 %r<12>;  → 12
    pattern = rf'\.reg \.{re.escape(reg_type)} %\w+<(\d+)>;'
    matches = re.findall(pattern, ptx)
    if not matches:
        return 0
    return sum(int(m) for m in matches)


VECTOR_ADD_SRC = """
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"""


def test_vector_add_compact_b32():
    """vector_add should declare <= 12 .b32 registers (not wasteful sparse IDs)."""
    ptx = _ptx(VECTOR_ADD_SRC)
    count = _count_regs(ptx, 'b32')
    assert count <= 12, \
        f"Expected <= 12 .b32 registers for vector_add, got {count}. PTX:\n{ptx}"


def test_vector_add_compact_b64():
    """vector_add should declare <= 12 .b64 registers."""
    ptx = _ptx(VECTOR_ADD_SRC)
    count = _count_regs(ptx, 'b64')
    assert count <= 12, \
        f"Expected <= 12 .b64 registers for vector_add, got {count}. PTX:\n{ptx}"


REGISTER_PRESSURE_SRC = """
__global__ void register_pressure(float *out, float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float a = in[tid * 8 + 0];
        float b = in[tid * 8 + 1];
        float c = in[tid * 8 + 2];
        float d = in[tid * 8 + 3];
        float e = in[tid * 8 + 4];
        float f = in[tid * 8 + 5];
        float g = in[tid * 8 + 6];
        float h = in[tid * 8 + 7];
        float sum  = a + b + c + d + e + f + g + h;
        float prod = a * b + c * d + e * f + g * h;
        float cross = (a + c + e + g) * (b + d + f + h);
        out[tid] = sum + prod + cross;
    }
}
"""


def test_register_pressure_compact_f32():
    """Kernel with 8 live floats + arithmetic should declare <= 16 .f32 registers."""
    ptx = _ptx(REGISTER_PRESSURE_SRC)
    count = _count_regs(ptx, 'f32')
    assert count <= 16, \
        f"Expected <= 16 .f32 registers for register_pressure, got {count}. PTX:\n{ptx}"


SIMPLE_KERNEL_SRC = """
__global__ void simple(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}
"""


def test_no_huge_register_gaps():
    """Simple kernel should declare <= 10 .f32 registers (no sparse ID gaps)."""
    ptx = _ptx(SIMPLE_KERNEL_SRC)
    count = _count_regs(ptx, 'f32')
    assert count <= 10, \
        f"Expected <= 10 .f32 registers for simple kernel, got {count}. PTX:\n{ptx}"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_mixed_int_float_mul():
    """(float)a[tid] * b[tid] should emit mul.f32 (not integer multiply)."""
    src = """
__global__ void k(float *out, int *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (float)a[tid] * b[tid];
    }
}
"""
    ptx = _ptx(src)
    assert 'mul.f32' in ptx, \
        f"Expected 'mul.f32' for int-to-float promoted multiply, got:\n{ptx}"


def test_printf_four_args_local_size():
    """printf with 4 int args should allocate .local .b8 _valist with >= 32 bytes."""
    src = """
__global__ void k(int *data) {
    if (threadIdx.x == 0) {
        int a = data[0];
        int b = data[1];
        int c = data[2];
        int d = data[3];
        printf("a=%d b=%d c=%d d=%d\\n", a, b, c, d);
    }
}
"""
    ptx = _ptx(src)
    # Find the valist allocation size: .local .align 8 .b8 _valist_0[32]
    m = re.search(r'\.local \.align 8 \.b8 \S+\[(\d+)\]', ptx)
    assert m is not None, \
        f"Expected .local .b8 valist in PTX for 4-arg printf, got:\n{ptx}"
    size = int(m.group(1))
    assert size >= 32, \
        f"Expected >= 32 bytes for 4-arg valist (4 * 8), got {size}"
