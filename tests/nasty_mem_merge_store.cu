// Value computed in both arms of an if/else, single store after merge.
// Tests that the stored value is the post-merge one.
__global__ void nasty_mem_merge_store(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float result;
        if (a[tid] > b[tid]) {
            result = a[tid] - b[tid];
        } else {
            result = b[tid] - a[tid];
        }
        out[tid] = result;
    }
}
