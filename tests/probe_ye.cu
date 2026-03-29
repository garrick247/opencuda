// Probe: #ifdef/#ifndef/#else/#endif, multi-line macro, recursive macro
// (double-expansion), __builtin_popcount/__builtin_clz, atomicMin/atomicMax
// on unsigned, texture-free bilinear interpolation, scan with warp shfl,
// prefix-or/prefix-and via shfl, tiled matmul with register blocking.

// ------------------------------------------------------------------
// #ifdef / #ifndef / #else / #endif.

#define FEATURE_A
// #define FEATURE_B  // intentionally not defined

#ifdef FEATURE_A
#define COEFF_A 2
#else
#define COEFF_A 1
#endif

#ifndef FEATURE_B
#define COEFF_B 10
#else
#define COEFF_B 20
#endif

__global__ void ifdef_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] * COEFF_A + COEFF_B;
    // expects: in[tid] * 2 + 10
}

// ------------------------------------------------------------------
// Multi-line macro with continuation.

#define CLAMP(x, lo, hi) \
    ((x) < (lo) ? (lo) : \
     (x) > (hi) ? (hi) : (x))

__global__ void clamp_macro(int *out, int *in, int lo, int hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = CLAMP(in[tid], lo, hi);
}

// ------------------------------------------------------------------
// Bilinear interpolation (texture-free, register-based).

__device__ float bilinear(float q00, float q01, float q10, float q11,
                            float tx, float ty) {
    float r0 = q00 * (1.0f - tx) + q10 * tx;
    float r1 = q01 * (1.0f - tx) + q11 * tx;
    return r0 * (1.0f - ty) + r1 * ty;
}

__global__ void bilinear_kernel(float *out,
                                  float *q00, float *q01,
                                  float *q10, float *q11,
                                  float *tx,  float *ty, int n) {
    int tid = threadIdx.x;
    if (tid < n)
        out[tid] = bilinear(q00[tid], q01[tid],
                            q10[tid], q11[tid],
                            tx[tid],  ty[tid]);
}

// ------------------------------------------------------------------
// Warp scan (prefix sum) using shfl_up_sync — same as probe_xt but
// with explicit per-step conditional + running accumulator.

__global__ void warp_scan_v2(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int lane = threadIdx.x & 31;
    int v = in[gid];
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if (lane >= d) v += t;
    }
    out[gid] = v;
}

// ------------------------------------------------------------------
// Prefix-OR via shfl_down_sync (not just sum).

__global__ void prefix_or(unsigned *out, unsigned *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = (gid < n) ? in[gid] : 0u;
    // reduce
    v |= __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v |= __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v |= __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v |= __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v |= __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if ((threadIdx.x & 31) == 0 && gid < n) out[gid / 32] = v;
}

// ------------------------------------------------------------------
// atomicMin / atomicMax on unsigned.

__global__ void atomic_uminmax(unsigned *umin_out, unsigned *umax_out,
                                  unsigned *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicMin(umin_out, in[gid]);
        atomicMax(umax_out, in[gid]);
    }
}

// ------------------------------------------------------------------
// 2×2 register-blocked matmul tile.

__global__ void matmul_2x2_tile(float *C, float *A, float *B, int K, int N) {
    int row = blockIdx.y * 2 + threadIdx.y;
    int col = blockIdx.x * 2 + threadIdx.x;
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    float a0, a1, b0, b1;
    for (int k = 0; k < K; k++) {
        a0 = A[(row*2  ) * K + k];
        a1 = A[(row*2+1) * K + k];
        b0 = B[k * N + col*2  ];
        b1 = B[k * N + col*2+1];
        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }
    C[(row*2  ) * N + col*2  ] = c00;
    C[(row*2  ) * N + col*2+1] = c01;
    C[(row*2+1) * N + col*2  ] = c10;
    C[(row*2+1) * N + col*2+1] = c11;
}

// ------------------------------------------------------------------
// __builtin_popcount / __builtin_clz (GCC builtins CUDA supports).

__global__ void builtin_ops(int *out_pop, int *out_clz,
                               unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_pop[tid] = __builtin_popcount(in[tid]);
        out_clz[tid] = __builtin_clz(in[tid]);
    }
}

// ------------------------------------------------------------------
// Integer log2 via __clz.

__device__ int ilog2(unsigned v) {
    if (v == 0) return -1;
    return 31 - __clz(v);
}

__global__ void ilog2_kernel(int *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = ilog2(in[tid]);
}
