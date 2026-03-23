// Parallel arrays of half, float, int32; load from each, cross-add, store.
// Tests mixed-width loads in one kernel.
__global__ void nasty_mem_mixed_load(half *hp, float *fp, int *ip, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float hv = (float)hp[tid];
        float fv = fp[tid];
        float iv = (float)ip[tid];
        out[tid] = hv + fv + iv;
    }
}
