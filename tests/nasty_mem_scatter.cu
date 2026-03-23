// Gather input indices from one array, scatter output to those positions.
// Tests non-contiguous address computation.
__global__ void nasty_mem_scatter(int *indices, float *data, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int idx = indices[tid];
        float val = data[tid];
        out[idx] = val * 2.0f;
    }
}
