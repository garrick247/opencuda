// Probe: C++ template-style macros, typename/template keywords that
// should parse without error (parsed as IDENT and skipped)

// Template function — parser should handle it (inline as regular function)
template<typename T>
__device__ T my_max(T a, T b) {
    return a > b ? a : b;
}

template<typename T>
__device__ T my_min(T a, T b) {
    return a < b ? a : b;
}

// Use template specialization — parser should handle instantiation as regular call
__global__ void template_max(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = my_max(a[tid], b[tid]);
    }
}

__global__ void template_minmax(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int clamped = my_max(my_min(v, 100), -100);
        out[tid] = clamped;
    }
}
