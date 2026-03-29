// Probe: __device__ __forceinline__ combination, __noinline__,
// function attributes that should be silently consumed

__device__ __forceinline__ float fast_rcp(float v) {
    return __frcp_rn(v);
}

__device__ __forceinline__ float fast_sqrt_rcp(float v) {
    return rsqrtf(v);
}

__global__ void forceinline_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid] + 1.0f;
        out[tid] = fast_rcp(v) + fast_sqrt_rcp(v);
    }
}

// __noinline__ hint
__device__ __noinline__ int expensive_fn(int x) {
    int r = x;
    for (int i = 0; i < 8; i++) {
        r = (r * 1664525 + 1013904223) & 0x7FFFFFFF;
    }
    return r;
}

__global__ void noinline_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = expensive_fn(in[tid]);
    }
}
