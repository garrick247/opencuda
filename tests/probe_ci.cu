// Probe: Patterns involving __device__ and __global__ in unexpected combinations
// - __device__ __forceinline__ function  
// - __device__ function with default parameter values (C++, should fail gracefully)
// - __device__ overloaded function names (C++ overloading)
// - Multiple kernels with same parameter names
// - __global__ with template-like macros

// __forceinline__ __device__ is already handled as KW_DEVICE
__device__ __forceinline__ float fast_inv(float x) {
    return 1.0f / x;
}

// Two kernels with identical param names but different bodies
__global__ void process_a(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] * 2.0f;
}

__global__ void process_b(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] + 1.0f;
}

__global__ void use_fast_inv(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fast_inv(in[tid] + 0.001f);
    }
}
