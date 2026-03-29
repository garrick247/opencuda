// Probe: global variable in complex lvalue/rvalue positions,
// multiple global arrays, conditional global array writes,
// global arrays in loops, and mixed global/shared patterns.

// ------------------------------------------------------------------
// Multiple global arrays in same kernel.

__device__ int g_a[32] = {0};
__device__ int g_b[32] = {0};
__device__ int g_c[32] = {0};

__global__ void multi_global_rw(int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        g_c[tid] = g_a[tid] + g_b[tid];
    }
}

__global__ void read_global_c(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        out[tid] = g_c[tid];
    }
}

// ------------------------------------------------------------------
// Global array write with conditional index.

__device__ int g_out_arr[64] = {0};

__global__ void cond_global_write(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int idx = (v > 0) ? tid : n - 1 - tid;
        if (idx < 64) {
            g_out_arr[idx] = v;
        }
    }
}

// ------------------------------------------------------------------
// Global array in loop: accumulate into global bins.

__device__ int g_buckets[8] = {0};

__global__ void fill_buckets(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int bucket = data[tid] & 7;
        atomicAdd(&g_buckets[bucket], 1);
    }
}

__global__ void read_buckets(int *out) {
    int tid = threadIdx.x;
    if (tid < 8) {
        out[tid] = g_buckets[tid];
    }
}

// ------------------------------------------------------------------
// Global scalar modified by multiple kernels.

__device__ float g_total = 0.0f;
__device__ int   g_count = 0;

__global__ void accumulate_global(float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(&g_total, data[tid]);  // note: atomicAdd on float global
        atomicAdd(&g_count, 1);
    }
}

__global__ void read_global_mean(float *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int cnt = g_count;
        float total = g_total;
        out[0] = (cnt > 0) ? total / (float)cnt : 0.0f;
    }
}

// ------------------------------------------------------------------
// Global lookup table used with a computed index.
// Tests that LUT reads with non-trivial indices work.

__device__ int g_fib[16] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};

__global__ void fib_lookup(int *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = (idx[tid] & 15);  // clamp to [0,15]
        out[tid] = g_fib[i];
    }
}

// ------------------------------------------------------------------
// Global array as both source and destination in same kernel.
// g_a[tid] += g_b[tid] — read-modify-write on separate global arrays.

__global__ void global_rmw(int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        g_a[tid] += g_b[tid];
    }
}
