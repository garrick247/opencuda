// Probe: C++ specific syntax that appears in CUDA device code
// - Constructor-style initialization: int x(5)
// - Range-based for (should fail gracefully)
// - auto keyword
// - nullptr
// - static_cast<T>

__global__ void static_cast_usage(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = static_cast<float>(in[tid]);
        out[tid] = v * 2.0f;
    }
}

__global__ void nullptr_check(int *out, int *ptr, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // nullptr comparison
        if (ptr != nullptr) {
            out[tid] = ptr[tid];
        } else {
            out[tid] = -1;
        }
    }
}
