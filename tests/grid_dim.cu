// Regression: gridDim.x/y/z must emit %nctaid.x/y/z PTX special registers.
// Standard CUDA pattern: compute global thread ID across multiple blocks.
__global__ void grid_dim_test(float *out, float *in, int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    if (tid < (unsigned int)n) {
        out[tid] = in[tid] * (float)stride;
    }
}
