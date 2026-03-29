// Probe: Unusual type declarations in function parameters
// - Parameter declared as void* (opaque)  
// - Parameter with anonymous struct type
// - Parameter with complex const qualifiers
// - Returning pointer to type (const vs non-const)

__device__ float array_sum(const float * const data, const int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

__device__ const float* find_max_ptr(const float *data, int n) {
    const float *best = data;
    for (int i = 1; i < n; i++) {
        if (data[i] > *best) {
            best = data + i;
        }
    }
    return best;
}

__global__ void const_ptr_test(float *out, const float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        float sum = array_sum(in, n);
        const float *max_ptr = find_max_ptr(in, n);
        out[0] = sum;
        out[1] = *max_ptr;
    }
}
