// Probe: multi-dim shared mem with runtime indexing, __syncwarp, __activemask,
// __ldg with various types (int2/float4 not used — pointer loads), texture-free
// gather pattern, parameter pack (variadics not in CUDA but multi-arg dispatch),
// __clzll/__popcll, 64-bit atomics (atomicAdd(ull*), atomicMax(ll*)),
// function returning struct, nested ternary in array index, __any_sync/__all_sync,
// and complex multi-level pointer dereference.

// ------------------------------------------------------------------
// __syncwarp and __activemask.

__global__ void syncwarp_test(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        __syncwarp();
        unsigned mask = __activemask();
        out[gid] = v + (int)mask;
    }
}

// ------------------------------------------------------------------
// __any_sync / __all_sync / __ballot_sync.

__global__ void vote_sync_test(int *out_any, int *out_all, int *out_ballot,
                                  int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int any  = __any_sync(0xFFFFFFFF, v > 0);
        int all  = __all_sync(0xFFFFFFFF, v > 0);
        unsigned bal = __ballot_sync(0xFFFFFFFF, v > 0);
        out_any[gid]    = any;
        out_all[gid]    = all;
        out_ballot[gid] = (int)bal;
    }
}

// ------------------------------------------------------------------
// __clzll / __popcll (64-bit count-leading-zeros / popcount).

__global__ void clzll_popcll(int *out_clz, int *out_pop,
                               long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        out_clz[tid] = __clzll(v);
        out_pop[tid] = __popcll(v);
    }
}

// ------------------------------------------------------------------
// 64-bit atomics: atomicAdd(unsigned long long *), atomicMax(long long *).

__global__ void atomic64(unsigned long long *sum, long long *maxval,
                          long long *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicAdd(sum, (unsigned long long)in[gid]);
        atomicMax(maxval, in[gid]);
    }
}

// ------------------------------------------------------------------
// Function returning a struct.

struct Pair { int lo, hi; };

__device__ struct Pair make_pair(int a, int b) {
    struct Pair p;
    p.lo = (a < b) ? a : b;
    p.hi = (a < b) ? b : a;
    return p;
}

__global__ void return_struct_kernel(int *out_lo, int *out_hi,
                                       int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Pair p = make_pair(a[tid], b[tid]);
        out_lo[tid] = p.lo;
        out_hi[tid] = p.hi;
    }
}

// ------------------------------------------------------------------
// Nested ternary in array index.

__global__ void ternary_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // read from clipped index
        int idx = (v < 0) ? 0 : (v >= n ? n - 1 : v);
        out[tid] = in[idx];
    }
}

// ------------------------------------------------------------------
// Multi-dim shared memory with runtime row/col indexing.

__global__ void shared_2d(float *out, float *in, int rows, int cols) {
    __shared__ float tile[16][16];
    int r = threadIdx.y;
    int c = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + c;
    if (gid < cols && r < rows) {
        tile[r][c] = in[r * cols + gid];
    }
    __syncthreads();
    if (gid < cols && r < rows) {
        out[r * cols + gid] = tile[r][c] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Pointer-to-pointer (double indirection): scatter via index array.

__global__ void gather_scatter(float *out, float *src, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        out[tid] = (i >= 0 && i < n) ? src[i] : 0.0f;
    }
}

// ------------------------------------------------------------------
// __mul64hi / __umul64hi (64-bit high-half multiply).

__global__ void mul64hi_kernel(long long *out_hi, unsigned long long *out_uhi,
                                  long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_hi[tid]  = __mul64hi(a[tid], b[tid]);
        out_uhi[tid] = __umul64hi((unsigned long long)a[tid],
                                   (unsigned long long)b[tid]);
    }
}

// ------------------------------------------------------------------
// Modulo-based hash function with bitwise ops.

__device__ unsigned hash32(unsigned x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__global__ void hash_kernel(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = hash32(in[tid]);
}
