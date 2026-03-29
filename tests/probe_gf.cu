// Probe: function overloading (same name, different param types)
// CUDA supports __device__ function overloading — parser needs to handle
// multiple functions with same name

__device__ float add_vals(float a, float b) { return a + b; }
__device__ int add_vals(int a, int b) { return a + b; }
__device__ double add_vals(double a, double b) { return a + b; }

__global__ void overload_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = add_vals(in[tid], in[(tid+1)%n]);
    }
}

__global__ void overload_int(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = add_vals(in[tid], in[(tid+1)%n]);
    }
}

// Default parameter values (C++ feature)
__device__ float weighted_add(float a, float b, float w) {
    return a * w + b * (1.0f - w);
}

__global__ void default_param_test(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = weighted_add(a[tid], b[tid], 0.5f);
    }
}
