// Liveness test: a value assigned in both branches (phi-like pattern).
// Linear scan must handle the two writes to r's physical register correctly.
__global__ void merge_reuse(float *out, float *a, float *b, int n, int flag) {
    int tid = threadIdx.x;
    if (tid < n) {
        float r;
        if (flag != 0) {
            r = a[tid] + b[tid];
        } else {
            r = a[tid] - b[tid];
        }
        out[tid] = r * r;
    }
}
