// Kernel with 5 pointer params (int*, float*, float*, int*, float*).
// Reads from 3, computes, writes to 2. Full ABI stress test.
__global__ void nasty_mem_multiptr(int *ia, float *fb, float *fc, int *id, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int a = ia[tid];
        float b = fb[tid];
        float c = fc[tid];
        id[tid] = a + (int)c;
        out[tid] = b + c + (float)a;
    }
}
