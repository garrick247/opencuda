__global__ void ldg_test(const float * __restrict__ in, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]);
    }
}
