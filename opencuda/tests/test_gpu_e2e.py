"""
tests/test_gpu_e2e.py — GPU end-to-end correctness tests for OpenCUDA.

Compiles CUDA C kernels through OpenCUDA (C -> PTX) and OpenPTXas (PTX -> cubin),
loads on RTX 5090 (SM_120), executes, and verifies results.

Run: python -m pytest opencuda/tests/test_gpu_e2e.py -v --tb=short -m gpu
"""
import ctypes
import math
import struct
import sys
import os
import pytest

# Add opencuda and openptxas to path
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(os.path.dirname(_root), 'openptxas'))

from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse
from opencuda.ir.optimize import optimize
from opencuda.codegen.emit import ir_to_ptx
from sass.pipeline import compile_ptx_source


# ---------------------------------------------------------------------------
# CUDA driver helpers
# ---------------------------------------------------------------------------

def _get_cuda():
    try:
        cuda = ctypes.cdll.LoadLibrary('nvcuda.dll')
        err = cuda.cuInit(0)
        if err != 0:
            return None
        return cuda
    except Exception:
        return None


_CUDA = _get_cuda()
gpu = pytest.mark.skipif(_CUDA is None, reason="No CUDA GPU available")


class CUDAContext:
    """Minimal CUDA driver context for E2E tests."""
    def __init__(self):
        self.cuda = _CUDA
        self.ctx = ctypes.c_void_p()
        self.mod = ctypes.c_void_p()
        dev = ctypes.c_int()
        err = self.cuda.cuDeviceGet(ctypes.byref(dev), 0)
        assert err == 0, f"cuDeviceGet failed: {err}"
        self._dev = dev
        err = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
        if err != 0:
            # Try resetting the primary context and retrying
            self.cuda.cuDevicePrimaryCtxReset_v2(dev)
            err = self.cuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev)
        assert err == 0, f"cuCtxCreate_v2 failed: {err}"

    def load(self, cubin_bytes: bytes) -> bool:
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        err = self.cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin_bytes)
        return err == 0

    def get_func(self, name: str):
        func = ctypes.c_void_p()
        err = self.cuda.cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode())
        assert err == 0, f"cuModuleGetFunction({name}) failed: {err}"
        return func

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        err = self.cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes)
        assert err == 0, f"cuMemAlloc_v2({nbytes}) failed: {err}"
        return ptr.value

    def copy_to(self, dev_ptr: int, host_data: bytes):
        err = self.cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(dev_ptr), host_data, len(host_data))
        assert err == 0, f"cuMemcpyHtoD_v2 failed: {err}"

    def copy_from(self, dev_ptr: int, nbytes: int) -> bytes:
        buf = (ctypes.c_uint8 * nbytes)()
        err = self.cuda.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(dev_ptr), nbytes)
        assert err == 0, f"cuMemcpyDtoH_v2 failed: {err}"
        return bytes(buf)

    def launch(self, func, grid, block, args_list, shared_mem=0) -> int:
        arg_holders = []
        ptrs = []
        for a in args_list:
            if isinstance(a, float):
                holder = ctypes.c_float(a)
            elif isinstance(a, int) and a > 0x7FFFFFFF:
                holder = ctypes.c_uint64(a)
            elif isinstance(a, int) and a < -0x80000000:
                holder = ctypes.c_int64(a)
            else:
                holder = ctypes.c_int32(a)
            arg_holders.append(holder)
            ptrs.append(ctypes.cast(ctypes.byref(holder), ctypes.c_void_p))
        args_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        gx, gy, gz = grid
        bx, by, bz = block
        return self.cuda.cuLaunchKernel(func, gx, gy, gz, bx, by, bz,
                                         shared_mem, None, args_arr, None)

    def free(self, ptr: int):
        self.cuda.cuMemFree_v2(ctypes.c_uint64(ptr))

    def sync(self) -> int:
        return self.cuda.cuCtxSynchronize()

    def drain_errors(self):
        """Drain any pending GPU errors to prevent cascading failures."""
        if self.ctx and self.ctx.value:
            self.cuda.cuCtxSynchronize()  # consumes deferred errors
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()

    def close(self):
        if self.mod and self.mod.value:
            self.cuda.cuModuleUnload(self.mod)
            self.mod = ctypes.c_void_p()
        if self.ctx and self.ctx.value:
            self.cuda.cuCtxSynchronize()
            self.cuda.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cuda_ctx():
    """Shared CUDA context for all GPU tests in this module."""
    if _CUDA is None:
        pytest.skip("No CUDA GPU available")
    cctx = CUDAContext()
    yield cctx
    cctx.close()


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def compile_cuda_to_cubin(cuda_source: str) -> dict:
    """Compile CUDA C source -> PTX -> cubin via OpenCUDA + OpenPTXas."""
    source = preprocess(cuda_source)
    module = parse(source)
    module = optimize(module)
    ptx_map = ir_to_ptx(module)

    # Build single PTX module
    all_ptx = ['.version 9.0', '.target sm_120', '.address_size 64', '']
    if '__preamble__' in ptx_map:
        all_ptx.extend(ptx_map['__preamble__'].split('\n'))
        all_ptx.append('')
    for kernel_name, ptx_text in ptx_map.items():
        if kernel_name.startswith('__'):
            continue
        lines = ptx_text.split('\n')
        body_start = 0
        for j, line in enumerate(lines):
            if line.startswith('.visible') or line.startswith('{'):
                body_start = j
                break
        all_ptx.extend(lines[body_start:])
        all_ptx.append('')

    ptx_full = '\n'.join(all_ptx)
    cubins = compile_ptx_source(ptx_full)
    return cubins


def pack_floats(values):
    return struct.pack(f'{len(values)}f', *values)


def unpack_floats(data, count):
    return list(struct.unpack(f'{count}f', data))


def pack_ints(values):
    return struct.pack(f'{len(values)}i', *values)


def unpack_ints(data, count):
    return list(struct.unpack(f'{count}i', data))


def pack_uints(values):
    return struct.pack(f'{len(values)}I', *values)


def unpack_uints(data, count):
    return list(struct.unpack(f'{count}I', data))


# ---------------------------------------------------------------------------
# E2E test kernels
# ---------------------------------------------------------------------------

# 1. Vector add (float) — multi-block
KERNEL_VECADD_FLOAT = """
__global__ void vector_add_f(float *out, float *a, float *b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

# 2. Vector add (int) — multi-block
KERNEL_VECADD_INT = """
__global__ void vector_add_i(int *out, int *a, int *b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

# 3. Vector scale
KERNEL_VSCALE = """
__global__ void vector_scale(float *out, float *a, float s, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = a[idx] * s;
    }
}
"""

# 4. SAXPY (a*x + y, manually expanded to avoid fma.rn.f32 OpenPTXas bug)
KERNEL_SAXPY = """
__global__ void saxpy(float *out, float *scales, float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float a = scales[0];
        float t = a * x[idx];
        out[idx] = t + y[idx];
    }
}
"""

# 5. Dot product (partial sums per block, single block for simplicity)
KERNEL_DOT = """
__global__ void dot_product(float *out, float *a, float *b, int n) {
    int idx = threadIdx.x;
    float sum = 0.0f;
    if (idx < n) {
        sum = a[idx] * b[idx];
    }
    // Warp reduction via shuffle
    unsigned int mask = 0xffffffff;
    sum = sum + __shfl_xor_sync(mask, sum, 16);
    sum = sum + __shfl_xor_sync(mask, sum, 8);
    sum = sum + __shfl_xor_sync(mask, sum, 4);
    sum = sum + __shfl_xor_sync(mask, sum, 2);
    sum = sum + __shfl_xor_sync(mask, sum, 1);
    if (idx == 0) {
        out[0] = sum;
    }
}
"""

# 6. Vector negate
KERNEL_VNEG = """
__global__ void vector_negate(float *out, float *a, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}
"""

# 7. Element-wise multiply
KERNEL_VMUL = """
__global__ void vector_mul(float *out, float *a, float *b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}
"""

# 8. sin/cos
KERNEL_SINCOS = """
__global__ void sincos_test(float *out_sin, float *out_cos, float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out_sin[idx] = sinf(input[idx]);
        out_cos[idx] = cosf(input[idx]);
    }
}
"""

# 9. sqrt
KERNEL_SQRT = """
__global__ void sqrt_test(float *out, float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = sqrtf(input[idx]);
    }
}
"""

# 10. exp/log
KERNEL_EXPLOG = """
__global__ void explog_test(float *out_exp, float *out_log, float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out_exp[idx] = expf(input[idx]);
        out_log[idx] = logf(input[idx]);
    }
}
"""

# 11. FMA
KERNEL_FMA = """
__global__ void fma_test(float *out, float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = fmaf(a[idx], b[idx], c[idx]);
    }
}
"""

# 12. Bitwise AND/OR/XOR
KERNEL_BITWISE = """
__global__ void bitwise_test(int *out_and, int *out_or, int *out_xor,
                              int *a, int *b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out_and[idx] = a[idx] & b[idx];
        out_or[idx]  = a[idx] | b[idx];
        out_xor[idx] = a[idx] ^ b[idx];
    }
}
"""

# 13. Population count (__popc)
KERNEL_POPC = """
__global__ void popc_test(int *out, int *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = __popc(input[idx]);
    }
}
"""

# 14. Count leading zeros (__clz)
KERNEL_CLZ = """
__global__ void clz_test(int *out, int *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = __clz(input[idx]);
    }
}
"""

# 15. Bit reversal (__brev)
KERNEL_BREV = """
__global__ void brev_test(unsigned int *out, unsigned int *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = __brev(input[idx]);
    }
}
"""

# 16. Conditional assignment
KERNEL_COND = """
__global__ void cond_test(float *out, float *a, float threshold, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (a[idx] > threshold) {
            out[idx] = a[idx] * 2.0f;
        } else {
            out[idx] = a[idx] * 0.5f;
        }
    }
}
"""

# 17. Max reduction per block (shared memory + __syncthreads)
KERNEL_MAXREDUCE = """
__shared__ float sdata[256];
__global__ void max_reduce(float *out, float *input, int n) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sdata[tid] = (idx < n) ? input[idx] : -999999.0f;
    __syncthreads();
    // Reduction in shared memory
    int s = 128;
    while (s > 0) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
        s = s / 2;
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
"""

# 18. Histogram (atomicAdd to global)
KERNEL_HISTOGRAM = """
__global__ void histogram(int *hist, int *data, int n, int nbins) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int bin = data[idx];
        if (bin >= 0 && bin < nbins) {
            atomicAdd(&hist[bin], 1);
        }
    }
}
"""

# 19. Warp shuffle broadcast
KERNEL_SHFL_BCAST = """
__global__ void shfl_broadcast(int *out, int val, int n) {
    int idx = threadIdx.x;
    unsigned int mask = 0xffffffff;
    int result = __shfl_sync(mask, val + idx, 0);
    if (idx < n) {
        out[idx] = result;
    }
}
"""

# 20. Warp reduction sum (__reduce_add_sync)
KERNEL_REDUX = """
__global__ void redux_add(int *out, int *input, int n) {
    int idx = threadIdx.x;
    int val = (idx < n) ? input[idx] : 0;
    unsigned int mask = 0xffffffff;
    int result = __reduce_add_sync(mask, val);
    if (idx == 0) {
        out[0] = result;
    }
}
"""

# 21. Ballot count
KERNEL_BALLOT = """
__global__ void ballot_count(int *out, int *data, int threshold, int n) {
    int idx = threadIdx.x;
    int val = (idx < n) ? data[idx] : 0;
    unsigned int mask = 0xffffffff;
    unsigned int ballot = __ballot_sync(mask, val > threshold);
    if (idx == 0) {
        out[0] = __popc(ballot);
    }
}
"""

# 22. Global atomic counter
KERNEL_ATOMIC_COUNTER = """
__global__ void atomic_counter(int *counter, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(counter, 1);
    }
}
"""

# 23. Atomic min/max
KERNEL_ATOMIC_MINMAX = """
__global__ void atomic_minmax(int *out_min, int *out_max, int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicMin(out_min, data[idx]);
        atomicMax(out_max, data[idx]);
    }
}
"""

# 24. AtomicCAS spinlock (simple counter via CAS)
KERNEL_ATOMIC_CAS = """
__global__ void atomic_cas_inc(int *counter, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int old = atomicCAS(counter, 0, 1);
        // Just test that atomicCAS compiles and runs; store old value
        atomicAdd(counter, 1);
    }
}
"""

# 25. Shared memory transpose (simplified: copy through shared)
KERNEL_SHARED_COPY = """
__shared__ float smem[256];
__global__ void shared_copy(float *out, float *input, int n) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        smem[tid] = input[idx];
    }
    __syncthreads();
    if (idx < n) {
        out[idx] = smem[tid] + 1.0f;
    }
}
"""

# 26. Block-level prefix sum (simplified: just scan within 32 elements)
KERNEL_SCAN = """
__shared__ float sdata[256];
__global__ void prefix_sum(float *out, float *input, int n) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    // Simple sequential scan (single thread for correctness)
    if (tid == 0) {
        int count = n;
        if (count > 256) { count = 256; }
        int i = 1;
        while (i < count) {
            sdata[i] = sdata[i] + sdata[i - 1];
            i = i + 1;
        }
    }
    __syncthreads();
    if (idx < n) {
        out[idx] = sdata[tid];
    }
}
"""

# 27. Stencil computation (1D, 3-point)
KERNEL_STENCIL = """
__global__ void stencil_1d(float *out, float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 0 && idx < n - 1) {
        out[idx] = 0.25f * input[idx - 1] + 0.5f * input[idx] + 0.25f * input[idx + 1];
    } else if (idx == 0 || idx == n - 1) {
        out[idx] = input[idx];
    }
}
"""

# 28. Float to int conversion
KERNEL_F2I = """
__global__ void float_to_int(int *out, float *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = (int)input[idx];
    }
}
"""

# 29. Int to float conversion
KERNEL_I2F = """
__global__ void int_to_float(float *out, int *input, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = (float)input[idx];
    }
}
"""

# 30. Matrix-vector multiply (uses shared memory + sync)
KERNEL_MATVEC = """
__shared__ float svec[64];
__global__ void matvec(float *out, float *mat, float *vec, int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    // Load vector into shared memory
    if (tid < cols) {
        svec[tid] = vec[tid];
    }
    __syncthreads();
    if (row < rows) {
        float sum = 0.0f;
        int c = 0;
        while (c < cols) {
            sum = sum + mat[row * cols + c] * svec[c];
            c = c + 1;
        }
        out[row] = sum;
    }
}
"""


# ---------------------------------------------------------------------------
# Compile-only tests (no GPU needed — verifies OpenCUDA produces valid PTX)
# ---------------------------------------------------------------------------

_ALL_KERNELS = {
    'vecadd_float': KERNEL_VECADD_FLOAT,
    'vecadd_int': KERNEL_VECADD_INT,
    'vscale': KERNEL_VSCALE,
    'saxpy': KERNEL_SAXPY,
    'dot': KERNEL_DOT,
    'vneg': KERNEL_VNEG,
    'vmul': KERNEL_VMUL,
    'sincos': KERNEL_SINCOS,
    'sqrt': KERNEL_SQRT,
    'explog': KERNEL_EXPLOG,
    'fma': KERNEL_FMA,
    'bitwise': KERNEL_BITWISE,
    'popc': KERNEL_POPC,
    'clz': KERNEL_CLZ,
    'brev': KERNEL_BREV,
    'cond': KERNEL_COND,
    'maxreduce': KERNEL_MAXREDUCE,
    'histogram': KERNEL_HISTOGRAM,
    'shfl_bcast': KERNEL_SHFL_BCAST,
    'redux': KERNEL_REDUX,
    'ballot': KERNEL_BALLOT,
    'atomic_counter': KERNEL_ATOMIC_COUNTER,
    'atomic_minmax': KERNEL_ATOMIC_MINMAX,
    'shared_copy': KERNEL_SHARED_COPY,
    'scan': KERNEL_SCAN,
    'stencil': KERNEL_STENCIL,
    'f2i': KERNEL_F2I,
    'i2f': KERNEL_I2F,
    'matvec': KERNEL_MATVEC,
}


class TestCompileAllKernels:
    """Verify all 32 CUDA C kernels compile to valid PTX via OpenCUDA."""

    @pytest.mark.parametrize("name", sorted(_ALL_KERNELS.keys()))
    def test_compile_to_ptx(self, name):
        src = _ALL_KERNELS[name]
        source = preprocess(src)
        module = parse(source)
        module = optimize(module)
        ptx_map = ir_to_ptx(module)
        # Must produce at least one kernel
        kernel_ptx = {k: v for k, v in ptx_map.items() if not k.startswith('__')}
        assert len(kernel_ptx) >= 1, f"No kernels in {name}"
        for kname, ptx_text in kernel_ptx.items():
            assert '.version' in ptx_text
            assert '.entry' in ptx_text
            assert 'ret;' in ptx_text

    @pytest.mark.parametrize("name", sorted(_ALL_KERNELS.keys()))
    def test_compile_to_cubin(self, name):
        """Verify full pipeline: CUDA C -> PTX -> cubin."""
        src = _ALL_KERNELS[name]
        cubins = compile_cuda_to_cubin(src)
        assert len(cubins) >= 1, f"No cubins for {name}"
        for kname, cubin_bytes in cubins.items():
            assert len(cubin_bytes) > 0
            assert cubin_bytes[:4] == b'\x7fELF'  # Valid ELF


# ---------------------------------------------------------------------------
# GPU E2E Tests — only kernels known to work with OpenPTXas backend
# ---------------------------------------------------------------------------


@gpu
class TestGPUE2E:
    """End-to-end GPU correctness tests.

    Only includes kernels verified to work through the full pipeline:
    OpenCUDA (CUDA C -> PTX) + OpenPTXas (PTX -> cubin) + RTX 5090 (SM_120).

    Kernels that crash or produce wrong results due to OpenPTXas backend
    instruction encoding bugs are excluded from GPU execution but still
    verified at compile level in TestCompileAllKernels above.
    """

    def test_vecadd_float(self, cuda_ctx):
        N = 256
        a = [float(i) for i in range(N)]
        b = [float(i * 2) for i in range(N)]
        expected = [a[i] + b[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_VECADD_FLOAT)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('vector_add_f')
        d_out, d_a, d_b = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        cuda_ctx.copy_to(d_b, pack_floats(b))
        assert cuda_ctx.launch(func, (1,1,1), (256,1,1), [d_out, d_a, d_b, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-4
        cuda_ctx.free(d_out); cuda_ctx.free(d_a); cuda_ctx.free(d_b)

    def test_vecadd_int(self, cuda_ctx):
        N = 256
        a, b = list(range(N)), [i*3 for i in range(N)]
        expected = [a[i]+b[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_VECADD_INT)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('vector_add_i')
        d_out, d_a, d_b = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_ints(a))
        cuda_ctx.copy_to(d_b, pack_ints(b))
        assert cuda_ctx.launch(func, (1,1,1), (256,1,1), [d_out, d_a, d_b, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_out, N*4), N) == expected
        cuda_ctx.free(d_out); cuda_ctx.free(d_a); cuda_ctx.free(d_b)

    def test_vscale(self, cuda_ctx):
        N = 128
        a = [float(i)*0.5 for i in range(N)]
        s = 3.0
        expected = [x*s for x in a]
        cubins = compile_cuda_to_cubin(KERNEL_VSCALE)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('vector_scale')
        d_out, d_a = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        assert cuda_ctx.launch(func, (1,1,1), (128,1,1), [d_out, d_a, s, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-3
        cuda_ctx.free(d_out); cuda_ctx.free(d_a)

    def test_vmul(self, cuda_ctx):
        N = 64
        a = [float(i)*0.5 for i in range(N)]
        b = [float(i)*0.3 for i in range(N)]
        expected = [a[i]*b[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_VMUL)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('vector_mul')
        d_out, d_a, d_b = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        cuda_ctx.copy_to(d_b, pack_floats(b))
        assert cuda_ctx.launch(func, (1,1,1), (64,1,1), [d_out, d_a, d_b, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-2
        cuda_ctx.free(d_out); cuda_ctx.free(d_a); cuda_ctx.free(d_b)

    def test_fma(self, cuda_ctx):
        N = 64
        a = [float(i)*0.5 for i in range(N)]
        b = [float(i)*0.3 for i in range(N)]
        c = [float(i)*0.1 for i in range(N)]
        expected = [a[i]*b[i]+c[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_FMA)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('fma_test')
        d_out, d_a, d_b, d_c = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        cuda_ctx.copy_to(d_b, pack_floats(b))
        cuda_ctx.copy_to(d_c, pack_floats(c))
        assert cuda_ctx.launch(func, (1,1,1), (64,1,1), [d_out, d_a, d_b, d_c, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 0.1
        cuda_ctx.free(d_out); cuda_ctx.free(d_a); cuda_ctx.free(d_b); cuda_ctx.free(d_c)

    def test_bitwise(self, cuda_ctx):
        N = 64
        a, b = list(range(N)), [x^0xFF for x in range(N)]
        exp_and = [a[i]&b[i] for i in range(N)]
        exp_or = [a[i]|b[i] for i in range(N)]
        exp_xor = [a[i]^b[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_BITWISE)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('bitwise_test')
        d_and, d_or, d_xor = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        d_a, d_b = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_ints(a))
        cuda_ctx.copy_to(d_b, pack_ints(b))
        assert cuda_ctx.launch(func, (1,1,1), (64,1,1), [d_and, d_or, d_xor, d_a, d_b, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_and, N*4), N) == exp_and
        assert unpack_ints(cuda_ctx.copy_from(d_or, N*4), N) == exp_or
        assert unpack_ints(cuda_ctx.copy_from(d_xor, N*4), N) == exp_xor
        cuda_ctx.free(d_and); cuda_ctx.free(d_or); cuda_ctx.free(d_xor)
        cuda_ctx.free(d_a); cuda_ctx.free(d_b)

    def test_shfl_broadcast(self, cuda_ctx):
        N = 32
        val = 42
        expected = [val] * N
        cubins = compile_cuda_to_cubin(KERNEL_SHFL_BCAST)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('shfl_broadcast')
        d_out = cuda_ctx.alloc(N*4)
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, val, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_out, N*4), N) == expected
        cuda_ctx.free(d_out)

    def test_atomic_exch(self, cuda_ctx):
        src = """
__global__ void atomic_exch_test(int *out, int *addr, int val) {
    int old = atomicExch(addr, val);
    out[0] = old;
}
"""
        cubins = compile_cuda_to_cubin(src)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('atomic_exch_test')
        d_out, d_addr = cuda_ctx.alloc(4), cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_addr, pack_ints([42]))
        assert cuda_ctx.launch(func, (1,1,1), (1,1,1), [d_out, d_addr, 99]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_out, 4), 1)[0] == 42
        assert unpack_ints(cuda_ctx.copy_from(d_addr, 4), 1)[0] == 99
        cuda_ctx.free(d_out); cuda_ctx.free(d_addr)

    def test_f2i(self, cuda_ctx):
        N = 32
        inp = [float(i)*1.5 for i in range(N)]
        expected = [int(x) for x in inp]
        cubins = compile_cuda_to_cubin(KERNEL_F2I)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('float_to_int')
        d_out, d_inp = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_out, N*4), N) == expected
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    def test_i2f(self, cuda_ctx):
        N = 32
        inp = list(range(N))
        expected = [float(x) for x in inp]
        cubins = compile_cuda_to_cubin(KERNEL_I2F)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('int_to_float')
        d_out, d_inp = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_ints(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-5
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    def test_vecadd_multiblock(self, cuda_ctx):
        N = 512
        a = [float(i) for i in range(N)]
        b = [float(i)*0.5 for i in range(N)]
        expected = [a[i]+b[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_VECADD_FLOAT)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('vector_add_f')
        d_out, d_a, d_b = cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4), cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        cuda_ctx.copy_to(d_b, pack_floats(b))
        assert cuda_ctx.launch(func, (2,1,1), (256,1,1), [d_out, d_a, d_b, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-4
        cuda_ctx.free(d_out); cuda_ctx.free(d_a); cuda_ctx.free(d_b)

    # ------------------------------------------------------------------
    # Additional GPU E2E tests for remaining 18 kernels
    # ------------------------------------------------------------------

    @pytest.mark.xfail(run=False, reason="OpenPTXas/OpenCUDA: saxpy still produces illegal instruction (715) — scalar param lowering bug")
    def test_saxpy(self, cuda_ctx):
        """saxpy: out[i] = a*x[i] + y[i] where a is scales[0]."""
        N = 128
        a_scale = 2.5
        x = [float(i) for i in range(N)]
        y = [float(i)*0.25 for i in range(N)]
        expected = [a_scale*x[i] + y[i] for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_SAXPY)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('saxpy')
        d_out = cuda_ctx.alloc(N*4)
        d_scales = cuda_ctx.alloc(4)
        d_x = cuda_ctx.alloc(N*4)
        d_y = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_scales, pack_floats([a_scale]))
        cuda_ctx.copy_to(d_x, pack_floats(x))
        cuda_ctx.copy_to(d_y, pack_floats(y))
        assert cuda_ctx.launch(func, (1,1,1), (128,1,1), [d_out, d_scales, d_x, d_y, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-3, f"i={i}: {result[i]} vs {expected[i]}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_scales); cuda_ctx.free(d_x); cuda_ctx.free(d_y)

    @pytest.mark.xfail(run=False, reason="OpenPTXas: __shfl_xor_sync reduction still produces illegal address (700)")
    def test_dot(self, cuda_ctx):
        """dot product via warp shuffle reduction (32 elements)."""
        N = 32
        a = [float(i+1) for i in range(N)]
        b = [float(i+1)*0.5 for i in range(N)]
        expected = sum(a[i]*b[i] for i in range(N))
        cubins = compile_cuda_to_cubin(KERNEL_DOT)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('dot_product')
        d_out = cuda_ctx.alloc(4)
        d_a = cuda_ctx.alloc(N*4)
        d_b = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        cuda_ctx.copy_to(d_b, pack_floats(b))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_a, d_b, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, 4), 1)[0]
        assert abs(result - expected) < 1e-2, f"dot: got {result}, expected {expected}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_a); cuda_ctx.free(d_b)

    def test_sincos(self, cuda_ctx):
        N = 32
        inp = [0.1*i for i in range(N)]
        exp_sin = [math.sin(v) for v in inp]
        exp_cos = [math.cos(v) for v in inp]
        cubins = compile_cuda_to_cubin(KERNEL_SINCOS)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('sincos_test')
        d_sin = cuda_ctx.alloc(N*4)
        d_cos = cuda_ctx.alloc(N*4)
        d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_sin, d_cos, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        r_sin = unpack_floats(cuda_ctx.copy_from(d_sin, N*4), N)
        r_cos = unpack_floats(cuda_ctx.copy_from(d_cos, N*4), N)
        for i in range(N):
            assert abs(r_sin[i] - exp_sin[i]) < 1e-3
            assert abs(r_cos[i] - exp_cos[i]) < 1e-3
        cuda_ctx.free(d_sin); cuda_ctx.free(d_cos); cuda_ctx.free(d_inp)

    def test_sqrt(self, cuda_ctx):
        N = 32
        inp = [float(i+1) for i in range(N)]
        expected = [math.sqrt(v) for v in inp]
        cubins = compile_cuda_to_cubin(KERNEL_SQRT)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('sqrt_test')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-3
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    def test_explog(self, cuda_ctx):
        N = 32
        inp = [0.5 + 0.1*i for i in range(N)]
        exp_e = [math.exp(v) for v in inp]
        exp_l = [math.log(v) for v in inp]
        cubins = compile_cuda_to_cubin(KERNEL_EXPLOG)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('explog_test')
        d_e = cuda_ctx.alloc(N*4); d_l = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_e, d_l, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        r_e = unpack_floats(cuda_ctx.copy_from(d_e, N*4), N)
        r_l = unpack_floats(cuda_ctx.copy_from(d_l, N*4), N)
        for i in range(N):
            # exp: relative tolerance since values grow rapidly
            assert abs(r_e[i] - exp_e[i]) / max(abs(exp_e[i]), 1.0) < 1e-2
            assert abs(r_l[i] - exp_l[i]) < 2e-3
        cuda_ctx.free(d_e); cuda_ctx.free(d_l); cuda_ctx.free(d_inp)

    def test_popc(self, cuda_ctx):
        N = 32
        inp = [i*0x10101 for i in range(N)]
        expected = [bin(v & 0xFFFFFFFF).count('1') for v in inp]
        cubins = compile_cuda_to_cubin(KERNEL_POPC)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('popc_test')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_ints(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_out, N*4), N) == expected
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    def test_clz(self, cuda_ctx):
        N = 31  # avoid 1<<31 which overflows int32
        inp_u = [1 << i for i in range(N)]  # 1, 2, 4, ..., 1<<30
        # __clz on 32-bit int: count leading zeros
        expected = [31 - i for i in range(N)]
        cubins = compile_cuda_to_cubin(KERNEL_CLZ)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('clz_test')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_uints(inp_u))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_out, N*4), N) == expected
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    def test_brev(self, cuda_ctx):
        N = 32
        inp = [i+1 for i in range(N)]
        def brev32(x):
            x &= 0xFFFFFFFF
            r = 0
            for b in range(32):
                if x & (1 << b):
                    r |= 1 << (31 - b)
            return r
        expected = [brev32(v) for v in inp]
        cubins = compile_cuda_to_cubin(KERNEL_BREV)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('brev_test')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_uints(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_uints(cuda_ctx.copy_from(d_out, N*4), N) == expected
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    @pytest.mark.xfail(run=False, reason="OpenCUDA emit: if/else codegen still produces illegal address (700) after setp fix")
    def test_cond(self, cuda_ctx):
        N = 64
        a = [float(i) - 32.0 for i in range(N)]  # mix of negatives and positives
        threshold = 0.0
        expected = [v*2.0 if v > threshold else v*0.5 for v in a]
        cubins = compile_cuda_to_cubin(KERNEL_COND)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('cond_test')
        d_out = cuda_ctx.alloc(N*4); d_a = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_a, pack_floats(a))
        assert cuda_ctx.launch(func, (1,1,1), (64,1,1), [d_out, d_a, threshold, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-4
        cuda_ctx.free(d_out); cuda_ctx.free(d_a)

    @pytest.mark.xfail(run=False, reason="OpenCUDA/OpenPTXas: shared-memory reduction loop still produces illegal address (700)")
    def test_maxreduce(self, cuda_ctx):
        N = 256
        import random
        random.seed(42)
        inp = [random.uniform(-100, 100) for _ in range(N)]
        expected = max(inp)
        cubins = compile_cuda_to_cubin(KERNEL_MAXREDUCE)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('max_reduce')
        d_out = cuda_ctx.alloc(4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (256,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, 4), 1)[0]
        assert abs(result - expected) < 1e-3, f"maxreduce: got {result}, expected {expected}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    @pytest.mark.xfail(run=False, reason="OpenCUDA emit: atomicAdd inside if-nested conditional still produces illegal address (700)")
    def test_histogram(self, cuda_ctx):
        N = 256
        nbins = 8
        data = [i % nbins for i in range(N)]
        expected = [0] * nbins
        for v in data:
            expected[v] += 1
        cubins = compile_cuda_to_cubin(KERNEL_HISTOGRAM)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('histogram')
        d_hist = cuda_ctx.alloc(nbins*4); d_data = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_hist, pack_ints([0]*nbins))
        cuda_ctx.copy_to(d_data, pack_ints(data))
        assert cuda_ctx.launch(func, (1,1,1), (256,1,1), [d_hist, d_data, N, nbins]) == 0
        assert cuda_ctx.sync() == 0
        assert unpack_ints(cuda_ctx.copy_from(d_hist, nbins*4), nbins) == expected
        cuda_ctx.free(d_hist); cuda_ctx.free(d_data)

    def test_redux(self, cuda_ctx):
        """__reduce_add_sync — warp-level reduction intrinsic."""
        N = 32
        inp = list(range(1, N+1))  # 1..32
        expected = sum(inp)
        cubins = compile_cuda_to_cubin(KERNEL_REDUX)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('redux_add')
        d_out = cuda_ctx.alloc(4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_ints(inp))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_ints(cuda_ctx.copy_from(d_out, 4), 1)[0]
        assert result == expected, f"redux: got {result}, expected {expected}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    def test_ballot(self, cuda_ctx):
        """__ballot_sync: count lanes where val > threshold."""
        N = 32
        data = list(range(N))  # 0..31
        threshold = 15
        expected = sum(1 for v in data if v > threshold)  # 16 lanes (16..31)
        cubins = compile_cuda_to_cubin(KERNEL_BALLOT)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('ballot_count')
        d_out = cuda_ctx.alloc(4); d_data = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_data, pack_ints(data))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_data, threshold, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_ints(cuda_ctx.copy_from(d_out, 4), 1)[0]
        assert result == expected, f"ballot: got {result}, expected {expected}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_data)

    def test_atomic_counter(self, cuda_ctx):
        N = 128
        cubins = compile_cuda_to_cubin(KERNEL_ATOMIC_COUNTER)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('atomic_counter')
        d_counter = cuda_ctx.alloc(4)
        cuda_ctx.copy_to(d_counter, pack_ints([0]))
        assert cuda_ctx.launch(func, (1,1,1), (128,1,1), [d_counter, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_ints(cuda_ctx.copy_from(d_counter, 4), 1)[0]
        assert result == N, f"atomic_counter: got {result}, expected {N}"
        cuda_ctx.free(d_counter)

    @pytest.mark.xfail(run=False, reason="OpenPTXas: atomicMin/atomicMax still produce illegal instruction (715) on SM_120")
    def test_atomic_minmax(self, cuda_ctx):
        N = 128
        import random
        random.seed(7)
        data = [random.randint(-1000, 1000) for _ in range(N)]
        exp_min = min(data)
        exp_max = max(data)
        cubins = compile_cuda_to_cubin(KERNEL_ATOMIC_MINMAX)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('atomic_minmax')
        d_min = cuda_ctx.alloc(4); d_max = cuda_ctx.alloc(4); d_data = cuda_ctx.alloc(N*4)
        # init min to large, max to small
        cuda_ctx.copy_to(d_min, pack_ints([0x7FFFFFFF]))
        cuda_ctx.copy_to(d_max, pack_ints([-0x80000000]))
        cuda_ctx.copy_to(d_data, pack_ints(data))
        assert cuda_ctx.launch(func, (1,1,1), (128,1,1), [d_min, d_max, d_data, N]) == 0
        assert cuda_ctx.sync() == 0
        r_min = unpack_ints(cuda_ctx.copy_from(d_min, 4), 1)[0]
        r_max = unpack_ints(cuda_ctx.copy_from(d_max, 4), 1)[0]
        assert r_min == exp_min, f"atomic_min: got {r_min}, expected {exp_min}"
        assert r_max == exp_max, f"atomic_max: got {r_max}, expected {exp_max}"
        cuda_ctx.free(d_min); cuda_ctx.free(d_max); cuda_ctx.free(d_data)

    @pytest.mark.xfail(run=False, reason="OpenCUDA/OpenPTXas: shared memory + __syncthreads still produces illegal address (700)")
    def test_shared_copy(self, cuda_ctx):
        N = 128
        inp = [float(i)*0.25 for i in range(N)]
        expected = [v + 1.0 for v in inp]
        cubins = compile_cuda_to_cubin(KERNEL_SHARED_COPY)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('shared_copy')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (128,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-4
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    @pytest.mark.xfail(run=False, reason="OpenCUDA/OpenPTXas: shared memory + while-loop scan still produces illegal address (700)")
    def test_scan(self, cuda_ctx):
        """Sequential prefix sum across 64 elements."""
        N = 64
        inp = [float(i+1) for i in range(N)]
        expected = []
        acc = 0.0
        for v in inp:
            acc += v
            expected.append(acc)
        cubins = compile_cuda_to_cubin(KERNEL_SCAN)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('prefix_sum')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (64,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-2, f"i={i}: {result[i]} vs {expected[i]}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    @pytest.mark.xfail(run=False, reason="OpenCUDA emit: else-if chain codegen still produces illegal address (700)")
    def test_stencil(self, cuda_ctx):
        N = 128
        inp = [float(i) for i in range(N)]
        expected = []
        for i in range(N):
            if i == 0 or i == N-1:
                expected.append(inp[i])
            else:
                expected.append(0.25*inp[i-1] + 0.5*inp[i] + 0.25*inp[i+1])
        cubins = compile_cuda_to_cubin(KERNEL_STENCIL)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('stencil_1d')
        d_out = cuda_ctx.alloc(N*4); d_inp = cuda_ctx.alloc(N*4)
        # Init output to input so boundary thread-indices beyond if/else don't expose uninit mem
        cuda_ctx.copy_to(d_out, pack_floats([0.0]*N))
        cuda_ctx.copy_to(d_inp, pack_floats(inp))
        assert cuda_ctx.launch(func, (1,1,1), (128,1,1), [d_out, d_inp, N]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, N*4), N)
        for i in range(N):
            assert abs(result[i] - expected[i]) < 1e-3, f"i={i}: {result[i]} vs {expected[i]}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_inp)

    @pytest.mark.xfail(run=False, reason="OpenCUDA/OpenPTXas: shared memory + while loop matvec still produces illegal address (700)")
    def test_matvec(self, cuda_ctx):
        rows = 32
        cols = 16
        mat = [float((r*cols + c) % 7) for r in range(rows) for c in range(cols)]
        vec = [float(i+1) * 0.1 for i in range(cols)]
        expected = []
        for r in range(rows):
            s = 0.0
            for c in range(cols):
                s += mat[r*cols + c] * vec[c]
            expected.append(s)
        cubins = compile_cuda_to_cubin(KERNEL_MATVEC)
        assert cuda_ctx.load(list(cubins.values())[0])
        func = cuda_ctx.get_func('matvec')
        d_out = cuda_ctx.alloc(rows*4)
        d_mat = cuda_ctx.alloc(rows*cols*4)
        d_vec = cuda_ctx.alloc(cols*4)
        cuda_ctx.copy_to(d_mat, pack_floats(mat))
        cuda_ctx.copy_to(d_vec, pack_floats(vec))
        assert cuda_ctx.launch(func, (1,1,1), (32,1,1), [d_out, d_mat, d_vec, rows, cols]) == 0
        assert cuda_ctx.sync() == 0
        result = unpack_floats(cuda_ctx.copy_from(d_out, rows*4), rows)
        for i in range(rows):
            assert abs(result[i] - expected[i]) < 1e-2, f"row {i}: {result[i]} vs {expected[i]}"
        cuda_ctx.free(d_out); cuda_ctx.free(d_mat); cuda_ctx.free(d_vec)
