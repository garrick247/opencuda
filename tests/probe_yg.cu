// Probe: __shfl_sync on int64/uint, printf in kernel, memset/memcpy on
// local arrays, struct-with-array passed by value to device func,
// integer division by constant (strength reduction), __noinline__,
// conditional expression as function argument, and deeply recursive
// inline (simulated via 4-level call chain).

// ------------------------------------------------------------------
// __shfl_sync on various types.

__global__ void shfl_int64(long long *out, long long *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    long long v = (gid < n) ? in[gid] : 0LL;
    // Reduce: sum of all lanes via XOR shuffle
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if ((threadIdx.x & 31) == 0 && gid < n) out[gid / 32] = v;
}

__global__ void shfl_unsigned(unsigned *out, unsigned *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = (gid < n) ? in[gid] : 0u;
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if ((threadIdx.x & 31) == 0 && gid < n) out[gid / 32] = v;
}

// ------------------------------------------------------------------
// struct with array member passed by value.

struct Hist8 { int bins[8]; };

__device__ int hist_max(struct Hist8 h) {
    int m = h.bins[0];
    for (int i = 1; i < 8; i++) if (h.bins[i] > m) m = h.bins[i];
    return m;
}

__global__ void hist_max_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Hist8 h;
        for (int i = 0; i < 8; i++) h.bins[i] = in[(tid * 8 + i) % n];
        out[tid] = hist_max(h);
    }
}

// ------------------------------------------------------------------
// Integer division by compile-time constant (strength reduction).

__global__ void div_by_const(int *out3, int *out7, int *out16,
                               int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out3[tid]  = v / 3;
        out7[tid]  = v / 7;
        out16[tid] = v / 16;
    }
}

// ------------------------------------------------------------------
// __noinline__ attribute.

__noinline__ __device__ int noinline_add(int a, int b) {
    return a + b + 1;
}

__global__ void noinline_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = noinline_add(a[tid], b[tid]);
}

// ------------------------------------------------------------------
// Conditional expression as function argument.

__device__ int triple(int x) { return x * 3; }

__global__ void cond_arg(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // conditional is the argument to triple
        out[tid] = triple(a[tid] > b[tid] ? a[tid] : b[tid]);
    }
}

// ------------------------------------------------------------------
// 4-level call chain (each function calls the next).

__device__ int level4(int x) { return x + 1; }
__device__ int level3(int x) { return level4(x) * 2; }
__device__ int level2(int x) { return level3(x) + level3(x - 1); }
__device__ int level1(int x) { return level2(x) - level2(x / 2); }

__global__ void chain4_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = level1(in[tid]);
}

// ------------------------------------------------------------------
// Prefix sum with divergent threads (not all lanes active).

__global__ void partial_warp_sum(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float v = (gid < n) ? in[gid] : 0.0f;
    // Only first 16 lanes contribute
    unsigned mask = 0x0000FFFF;
    if (lane < 16) {
        v += __shfl_xor_sync(mask, v, 8);
        v += __shfl_xor_sync(mask, v, 4);
        v += __shfl_xor_sync(mask, v, 2);
        v += __shfl_xor_sync(mask, v, 1);
        if (lane == 0 && gid < n) out[gid / 16] = v;
    }
}

// ------------------------------------------------------------------
// Interleaved global reads and shared-mem stores.

__global__ void interleaved_shared(float *out, float *in, int n) {
    __shared__ float buf[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) buf[tid] = in[gid];
    __syncthreads();
    // Reverse within block
    if (gid < n) out[gid] = buf[blockDim.x - 1 - tid];
}

// ------------------------------------------------------------------
// Multiple __device__ functions sharing the same parameter name (scope isolation).

__device__ int f_scope(int x) { return x * x; }
__device__ int g_scope(int x) { return x + x; }

__global__ void scope_iso_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        out[tid] = f_scope(x) + g_scope(x);
    }
}
