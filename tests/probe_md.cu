// Probe: Device function patterns
// - Single device function called once
// - Same device function called twice from same kernel (two call sites)
// - Device function calling another device function (chained calls)
// - Device function with multiple arguments of mixed types
// - Device function returning float

__device__ int square(int x) {
    return x * x;
}

__device__ int add_then_scale(int a, int b, int scale) {
    int sum = a + b;
    return sum * scale;
}

__device__ float normalize(float v, float lo, float hi) {
    return (v - lo) / (hi - lo);
}

// Calls square twice — two call sites, same function
__device__ int sum_of_squares(int a, int b) {
    return square(a) + square(b);
}

__global__ void dev_single_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = square(in[tid]);
    }
}

__global__ void dev_two_callsites(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = square(in[tid]);
        int b = square(in[tid] + 1);
        out[tid] = a + b;
    }
}

__global__ void dev_chained(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sum_of_squares calls square internally — chained
        out[tid] = sum_of_squares(in[tid], in[tid] + 1);
    }
}

__global__ void dev_mixed_args(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int result = add_then_scale(in[tid], tid, 3);
        out[tid] = result;
    }
}

__global__ void dev_float_return(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = normalize(v, 0.0f, 1.0f);
    }
}
