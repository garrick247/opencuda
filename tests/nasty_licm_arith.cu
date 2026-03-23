// LICM test: loop-invariant pure arithmetic.
__global__ void licm_arith_hoist(float *out, float *a, int n, float base, float step) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    for (int i = 0; i < n; i++) {
        float range  = base * step;   // both params, invariant -> HOIST
        float offset = base + step;   // both params, invariant -> HOIST
        out[tid + i] = a[tid + i] * range + offset;
    }
}

// Chained invariant arith: k2 must be hoisted before k4 can be recognized as invariant.
__global__ void licm_arith_chain(float *out, float *a, int n, float k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    for (int i = 0; i < n; i++) {
        float k2 = k * 2.0f;    // invariant -> HOIST (round 1)
        float k4 = k2 * 2.0f;   // invariant once k2 hoisted -> HOIST (round 2)
        out[tid + i] = a[tid + i] * k4;
    }
}
