// Load value, compute a condition, ONLY store if condition passes.
// Tests that guarded store uses correct address and doesn't emit store on wrong branch.
__global__ void nasty_mem_guarded_store(float *out, float *in, float threshold, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = in[tid];
        if (val > threshold) {
            out[tid] = val - threshold;
        }
    }
}
