// Probe: early return in device functions, return from inside loops,
// conditional early return, and complex return path merging.

// ------------------------------------------------------------------
// Device fn with early return (no return in all paths).

__device__ int find_first(const int *arr, int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

__global__ void search_kernel(int *out, int *data, int *queries, int n, int m) {
    int tid = threadIdx.x;
    if (tid < m) {
        out[tid] = find_first(data, n, queries[tid]);
    }
}

// ------------------------------------------------------------------
// Device fn: early return based on precondition.

__device__ float safe_sqrt(float x) {
    if (x < 0.0f) return 0.0f;  // early return
    return sqrtf(x);
}

__device__ float safe_log(float x) {
    if (x <= 0.0f) return -1e30f;  // early return
    return logf(x);
}

__global__ void safe_math(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = safe_sqrt(v);
        float b = safe_log(v + 1.0f);
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Return from inside nested loop.

__device__ int count_until_zero(const int *arr, int n) {
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4; j++) {
            if (arr[i * 4 + j] == 0) return cnt;
            cnt++;
        }
    }
    return cnt;
}

__global__ void count_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = count_until_zero(data + tid * 16, 4);
    }
}

// ------------------------------------------------------------------
// Device fn with multiple return types from branches (all same type).

__device__ float classify_value(float v, float thresh_lo, float thresh_hi) {
    if (v < thresh_lo) {
        return -1.0f * thresh_lo;
    }
    if (v > thresh_hi) {
        return thresh_hi;
    }
    float normalized = (v - thresh_lo) / (thresh_hi - thresh_lo);
    if (normalized < 0.5f) {
        return normalized * 2.0f;
    }
    return 2.0f - normalized * 2.0f;
}

__global__ void classify_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify_value(in[tid], -10.0f, 10.0f);
    }
}

// ------------------------------------------------------------------
// Kernel early return (not device fn — early return from kernel body).

__global__ void kernel_early_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;  // early kernel exit

    int v = in[tid];
    if (v == 0) {
        out[tid] = 0;
        return;  // second early exit
    }

    int r = 0;
    for (int i = 1; i <= v && i <= 10; i++) {
        if (v % i == 0) r++;
    }
    out[tid] = r;  // divisor count
}

// ------------------------------------------------------------------
// Return value used as argument immediately.

__device__ int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__global__ void chained_clamp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Nested calls: result of one used as arg to next
        int v = in[tid];
        int r = clamp_i(clamp_i(v, -100, 100) * 2, -50, 50);
        out[tid] = r;
    }
}
