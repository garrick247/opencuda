// Probe: multi-level device function calls with early returns,
// recursive-style depth (DAG, not true recursion), and complex
// return value flows through multiple call levels.

// ------------------------------------------------------------------
// Two-level call chain with early return in inner function.

__device__ int safe_div(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

__device__ int norm_div(int a, int b, int limit) {
    int d = safe_div(a, b);
    if (d > limit) return limit;
    if (d < -limit) return -limit;
    return d;
}

__global__ void two_level_early_return(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = norm_div(a[tid], b[tid], 100);
    }
}

// ------------------------------------------------------------------
// Three-level call chain.

__device__ float safe_sqrt(float v) {
    if (v < 0.0f) return 0.0f;
    return __sqrtf(v);
}

__device__ float rms2(float a, float b) {
    return safe_sqrt(a*a + b*b);
}

__device__ float rms3(float a, float b, float c) {
    float ab = rms2(a, b);
    return rms2(ab, c);
}

__global__ void three_level_chain(float *out, float *x, float *y, float *z, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = rms3(x[tid], y[tid], z[tid]);
    }
}

// ------------------------------------------------------------------
// Device function with conditional early return AND normal return.

__device__ int categorize(int v) {
    if (v < -100) return -2;
    if (v < 0)    return -1;
    if (v == 0)   return 0;
    if (v < 100)  return 1;
    return 2;
}

__global__ void categorize_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = categorize(in[tid]);
    }
}

// ------------------------------------------------------------------
// Mutual use: two device functions each calling a third.

__device__ int clamp_lo(int v, int lo) {
    return (v < lo) ? lo : v;
}

__device__ int clamp_hi(int v, int hi) {
    return (v > hi) ? hi : v;
}

__device__ int clamp_both(int v, int lo, int hi) {
    return clamp_hi(clamp_lo(v, lo), hi);
}

__global__ void fan_out_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = clamp_both(in[tid], -50, 50);
    }
}

// ------------------------------------------------------------------
// Device function called multiple times with different args.

__device__ float weighted_sum(float a, float wa, float b, float wb) {
    return a * wa + b * wb;
}

__global__ void multi_call(float *out, float *p, float *q, float *r, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s1 = weighted_sum(p[tid], 0.5f, q[tid], 0.5f);
        float s2 = weighted_sum(q[tid], 0.3f, r[tid], 0.7f);
        float s3 = weighted_sum(s1, 0.4f, s2, 0.6f);
        out[tid] = s3;
    }
}

// ------------------------------------------------------------------
// Device function whose result is used in a conditional.

__device__ int is_prime_small(int n) {
    if (n < 2)  return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void prime_filter(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = is_prime_small(v % 100) ? v : 0;
    }
}

// ------------------------------------------------------------------
// Result of device function used as array index.

__device__ int bucket(int v, int nbuckets) {
    if (v < 0) return 0;
    int b = v % nbuckets;
    return b;
}

__global__ void bucket_count(int *counts, int *in, int n, int nb) {
    int tid = threadIdx.x;
    if (tid < n) {
        int b = bucket(in[tid], nb);
        atomicAdd(&counts[b], 1);
    }
}
