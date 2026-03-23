// LICM test: loop-invariant type conversions inside loop bodies.
// (float)scale and (float)n never change across iterations — should be hoisted.
__global__ void licm_cvt_hoist(float *out, int *a, int n, int scale) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    for (int i = 0; i < n; i++) {
        float fscale = (float)scale;   // param, loop-invariant -> HOIST
        float fn     = (float)n;       // param, loop-invariant -> HOIST
        out[tid + i] = a[tid + i] * fscale + fn;
    }
}

// Chain: fx + fy is invariant once fx and fy are hoisted.
__global__ void licm_cvt_chain(float *out, int *a, int n, int x, int y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    for (int i = 0; i < n; i++) {
        float fx  = (float)x;    // invariant -> HOIST
        float fy  = (float)y;    // invariant -> HOIST
        float sum = fx + fy;     // both operands invariant after above hoists -> HOIST
        out[tid + i] = a[tid + i] + sum;
    }
}
