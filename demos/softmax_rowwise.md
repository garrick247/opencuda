# Row-wise Softmax — end-to-end open GPU compilation

**Source:** `softmax_rowwise.cu` (hand-written CUDA C)
**Pipeline:** OpenCUDA (Python) → OpenPTXas (Python) → SM_120 cubin → RTX 5090
**NVIDIA compilers used:** none.

## The kernel

A classic textbook softmax: each block computes softmax of one row of
a `(rows × cols)` matrix.  Two-pass tree reduction in shared memory:

1. **Pass 1 — find row max.** Each thread loads one element, fills
   unused lanes with `-∞`, then tree-reduces to `smem[0]`.
2. **Pass 2 — compute `exp(x - max)`, sum the exponentials.** Tree
   reduce again for the denominator.
3. **Normalize.** Each thread writes `exp(x - max) / sum`.

```cuda
__global__ void softmax_rowwise(float *inp, float *out, int n_cols)
{
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    float x = 0.0f;
    if (tid < n_cols) x = inp[row * n_cols + tid];

    // Pass 1 — max reduction
    smem[tid] = (tid < n_cols) ? x : -1e30f;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s /= 2) {
        if (tid < s && smem[tid + s] > smem[tid])
            smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float row_max = smem[0]; __syncthreads();

    // Pass 2 — exp then sum reduction
    float ex = (tid < n_cols) ? expf(x - row_max) : 0.0f;
    smem[tid] = ex; __syncthreads();
    for (int s = BLOCK/2; s > 0; s /= 2) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid < n_cols) out[row * n_cols + tid] = ex / smem[0];
}
```

## What each stage does

| Stage | Responsibility | Key features exercised |
|-------|---------------|------------------------|
| **OpenCUDA** (`python -m opencuda`) | Parse C, emit SM_120 PTX | `__shared__` arrays, `__syncthreads`, `expf → ex2.approx.f32 * log2(e)`, for-loops, conditional writes |
| **OpenPTXas** (`sass.pipeline`) | PTX → SASS + ELF cubin | FFMA, FMUL, FMNMX, `bar.sync`, shared-memory loads/stores, LDG, STG, scoreboard + rbar/wdep, LDCU.64 param loads |
| **CUDA driver** | Load cubin, launch kernel | — |

`expf` lowers through OpenCUDA as `fma (multiplier = 0f3FB8AA3B = log2(e))`
then `ex2.approx.f32` (one `MUFU.EX2` instruction on SM_120). The
OpenPTXas encoder has ground-truth bytes for `MUFU.EX2` from ptxas,
so the emitted cubin is byte-compatible with ptxas at the single-
instruction level.

## GPU results (RTX 5090, SM_120)

Compared against `scipy.special.softmax` / numpy-stable reference.

```
 rows  cols   correct     max_err    mean_err
--------------------------------------------------
    4    16      PASS    2.98e-08    3.75e-09
    8    64      PASS    1.49e-08    1.60e-09
   16   128      PASS    7.45e-09    6.01e-10
   32   256      PASS    7.45e-09    3.02e-10
  128   256      PASS    1.49e-08    3.03e-10
--------------------------------------------------
OVERALL: 5/5
```

Max absolute error ≈ 3e-8, which is at the floor of FP32 precision
(2^-23 ≈ 1.2e-7; softmax intermediate quantities compress to smaller
values so per-element error is smaller).  The only numerical
difference vs ptxas is in `ex2.approx.f32`'s hardware rounding, and
that's identical between our cubin and ptxas's because the
underlying MUFU instruction comes from the GPU, not from either
compiler.

## Reproduce

```bash
cd opencuda
python -m opencuda demos/softmax_rowwise.cu --emit-ptx --out demos/softmax_rowwise.ptx
cd ../openptxas
python -c "import sys; sys.path.insert(0, '.'); \
    from sass.pipeline import compile_ptx_source; \
    ptx = open('../opencuda/demos/softmax_rowwise.ptx').read(); \
    open('../opencuda/demos/softmax_rowwise.cubin', 'wb').write(\
        compile_ptx_source(ptx)['softmax_rowwise'])"
cd ../opencuda
python demos/softmax_rowwise_run.py
```

## Why this matters

Softmax is the numerical core of every attention layer in modern
LLMs.  Running it end-to-end through an **open-source Python**
CUDA C compiler (OpenCUDA) and PTX assembler (OpenPTXas), with
output correct to FP32 precision against a numpy reference, is a
demonstration that the stack handles real ML primitives — not just
synthetic micro-kernels.

The **Forge proof layer** is not used in this demo because Forge's
standard library doesn't yet expose f32 GPU types (`span<f32>` and
the `expf` intrinsic).  When that lands, the same kernel written in
`.fg` will have its shared-memory indices and bounds checks
discharged by Z3 at compile time — adding a formal proof layer on
top of the end-to-end runtime-verified pipeline shown here.

## Artifacts

- `softmax_rowwise.cu` — CUDA C source (70 lines)
- `softmax_rowwise.ptx` — OpenCUDA-emitted PTX 8.8
- `softmax_rowwise.cubin` — OpenPTXas-emitted SM_120 cubin
- `softmax_rowwise_run.py` — GPU harness, correctness check
