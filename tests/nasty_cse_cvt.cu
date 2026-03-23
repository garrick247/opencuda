// Tests repeated conversion CSE: same source converted to same type.
__global__ void cvt_dedup(float *out, int *a, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int v = a[tid];
    // The same int->float conversion appears three times; should CSE to one.
    float f1 = (float)v * 1.0f;
    float f2 = (float)v * 2.0f;
    float f3 = (float)v * 3.0f;
    out[tid] = f1 + f2 + f3;
}

__global__ void addr_widen_dedup(float *out, float *a, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    // Multiple loads from the same base — address widening should CSE.
    float v0 = a[tid];
    float v1 = a[tid + 1];
    out[tid] = v0 + v1;
}
