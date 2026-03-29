// Probe: More unusual but valid CUDA patterns
// - __device__ function with no return (void return type, no return stmt)
// - __device__ function with return inside nested if/else
// - __device__ function that calls another with same param name
// - Ternary in function argument position

__device__ void fill_range(float *arr, int start, int end, float val) {
    for (int i = start; i < end; i++) {
        arr[i] = val;
    }
    // implicit void return
}

__device__ float conditional_reciprocal(float x, float epsilon) {
    if (x > epsilon) {
        return 1.0f / x;
    } else if (x < -epsilon) {
        return 1.0f / x;
    } else {
        return 0.0f;
    }
}

__device__ float safe_div(float a, float b) {
    return conditional_reciprocal(b, 1e-6f) * a;
}

__global__ void complex_divide(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Ternary as function argument
        float result = safe_div(a[tid], (b[tid] != 0.0f) ? b[tid] : 1.0f);
        out[tid] = result;
    }
}
