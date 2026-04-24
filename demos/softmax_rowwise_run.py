"""Softmax demo — end-to-end verified via OpenCUDA + OpenPTXas.

Compiles demos/softmax_rowwise.cu through the entire open Python
stack (no nvcc, no ptxas), runs on RTX 5090, and checks the output
against numpy's softmax reference for multiple row widths.
"""
import ctypes, os
import numpy as np

CUBIN = os.path.abspath(os.path.join(os.path.dirname(__file__), 'softmax_rowwise.cubin'))

cuda = ctypes.CDLL("nvcuda.dll")
cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]

def ck(e, msg=""):
    if e != 0:
        raise RuntimeError(f"{msg}: cuda err {e}")

ck(cuda.cuInit(0))
DEV = ctypes.c_int(); ck(cuda.cuDeviceGet(ctypes.byref(DEV), 0))


def softmax_cpu(x, axis=-1):
    # numerically-stable reference
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)


# Row-wise softmax at several sizes; BLOCK=256 is fixed in the kernel.
shapes = [
    (4, 16),      # 4 rows × 16 cols
    (8, 64),
    (16, 128),
    (32, 256),
    (128, 256),   # 128 rows × 256 cols = full blocks
]

print(f"{'rows':>5} {'cols':>5}  {'correct':>8}  {'max_err':>10}  {'mean_err':>10}")
print("-" * 50)
ok = 0
total = 0
for (R, C) in shapes:
    ctx = ctypes.c_void_p(); ck(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, DEV))
    mod = ctypes.c_void_p(); ck(cuda.cuModuleLoad(ctypes.byref(mod), CUBIN.encode()))
    fn = ctypes.c_void_p(); ck(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"softmax_rowwise"))

    np.random.seed(R * 31 + C)
    x = np.random.randn(R, C).astype(np.float32)
    y_gpu = np.zeros((R, C), dtype=np.float32)

    nbytes = R * C * 4
    d_in = ctypes.c_uint64(); d_out = ctypes.c_uint64()
    ck(cuda.cuMemAlloc_v2(ctypes.byref(d_in),  nbytes))
    ck(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes))
    ck(cuda.cuMemcpyHtoD_v2(d_in,  x.ctypes.data_as(ctypes.c_void_p), nbytes))
    ck(cuda.cuMemcpyHtoD_v2(d_out, y_gpu.ctypes.data_as(ctypes.c_void_p), nbytes))

    p_in  = ctypes.c_uint64(d_in.value)
    p_out = ctypes.c_uint64(d_out.value)
    p_n   = ctypes.c_int32(C)
    params = (ctypes.c_void_p * 3)(*[
        ctypes.cast(ctypes.pointer(x), ctypes.c_void_p)
        for x in [p_in, p_out, p_n]
    ])
    BLOCK = 256
    ck(cuda.cuLaunchKernel(fn, R, 1, 1, BLOCK, 1, 1,
                            0, ctypes.c_void_p(0), params, ctypes.c_void_p(0)))
    ck(cuda.cuCtxSynchronize())
    ck(cuda.cuMemcpyDtoH_v2(y_gpu.ctypes.data, d_out, nbytes))

    y_ref = softmax_cpu(x, axis=1)
    err = np.abs(y_gpu - y_ref)
    max_err = float(err.max())
    mean_err = float(err.mean())
    # ex2.approx.f32 is ~23-bit accurate; allow ~1e-6 relative tolerance
    correct = max_err < 1e-5
    total += 1
    if correct: ok += 1
    print(f"{R:>5} {C:>5}  {'PASS' if correct else 'FAIL':>8}  {max_err:>10.2e}  {mean_err:>10.2e}")

    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy_v2(ctx)

print("-" * 50)
print(f"OVERALL: {ok}/{total}")
