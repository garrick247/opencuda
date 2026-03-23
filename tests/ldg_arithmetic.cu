__global__ void ldg_arithmetic(float *out, const float * __restrict__ a,
                                const float * __restrict__ b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = __ldg(&a[tid]);
        float y = __ldg(&b[tid]);
        float r = x * y + x - y;
        if (r < 0.0f) r = -r;
        out[tid] = r;
    }
}
