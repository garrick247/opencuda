// Probe: Patterns that test PTX correctness of generated code
// - Condition on float NaN (unordered compare)
// - Integer min/max via conditional
// - Abs of int (no abs() intrinsic for int in PTX — must emit manual)
// - Multiply-accumulate pattern (fma)

__device__ float fma_chain(float a, float b, float c, float d) {
    // (a*b + c) * d
    return (a * b + c) * d;
}

__global__ void fma_test(float *out, float *a, float *b, float *c, float *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fma_chain(a[tid], b[tid], c[tid], d[tid]);
    }
}

__global__ void int_abs(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = (v < 0) ? -v : v;
    }
}

__global__ void int_minmax(int *out_min, int *out_max, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid];
        out_min[tid] = (va < vb) ? va : vb;
        out_max[tid] = (va > vb) ? va : vb;
    }
}

// NaN check: x != x is true only for NaN
__global__ void nan_check(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = (v != v) ? 1 : 0;  // NaN test
    }
}
