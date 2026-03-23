// Each iteration computes a DIFFERENT output address (idx * stride + base).
// Stresses address recomputation per-iteration.
__global__ void nasty_mem_loop_store(float *out, float *in, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = in[tid];
        for (int i = 0; i < 4; i++) {
            out[tid * stride + i] = val * (float)(i + 1);
        }
    }
}
