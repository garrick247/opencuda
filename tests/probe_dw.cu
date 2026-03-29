// Probe: Pointer arithmetic in complex contexts
// - Pointer used as array subscript base: *(p + n)
// - Pointer arithmetic result as array subscript: p[n-1]
// - Pointer in ternary: cond ? ptr_a : ptr_b
// - Pointer incremented by expression: p += stride

__global__ void ptr_subscript_variants(float *out, float *in, int n, int stride) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *base = in + tid;  // pointer arithmetic
        float v1 = *(base);      // deref
        float v2 = *(base + stride < base + n ? base + stride : base + n - 1 - base + base);
        // Simpler version:
        float *p = in;
        p += tid;  // pointer += scalar
        float v3 = p[0];
        out[tid] = v1 + v3;
    }
}

// Pointer in conditional expression
__global__ void ptr_ternary(float *out, float *a, float *b, int *mask, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *chosen = mask[tid] ? a : b;
        out[tid] = chosen[tid];
    }
}

// Strided access via pointer advance
__global__ void strided_ptr(float *out, float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = in + tid;
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            if (p < in + n) {
                sum += *p;
                p += stride;
            }
        }
        out[tid] = sum;
    }
}
