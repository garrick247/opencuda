// LICM safety: loop-dependent and side-effecting ops must NOT be hoisted.
__global__ void licm_no_hoist_loop_var(float *out, float *a, int n, float k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    for (int i = 0; i < n; i++) {
        float fi = (float)i;      // i changes each iteration -> NOT hoistable
        out[tid + i] = a[tid + i] * fi * k;
    }
}

__global__ void licm_no_hoist_memory(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    for (int i = 0; i < n; i++) {
        float v = a[i];    // load — side-effecting -> NOT hoistable
        b[i]    = v;       // store — side-effecting -> NOT hoistable
        out[i]  = v;
    }
}
