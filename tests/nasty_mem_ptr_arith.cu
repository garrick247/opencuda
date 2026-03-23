// Pointer arithmetic under branches: compute two different address offsets
// based on a condition, use the result for load then store.
__global__ void nasty_mem_ptr_arith(float *evens, float *odds, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val;
        if (tid % 2 == 0) {
            val = evens[tid / 2];
        } else {
            val = odds[tid / 2];
        }
        out[tid] = val * 1.5f;
    }
}
