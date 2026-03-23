__device__ float categorize(float x) {
    if (x < -1.0f) return -2.0f;
    if (x < 0.0f)  return -1.0f;
    if (x < 1.0f)  return x;
    return 2.0f;
}

__global__ void inline_printf_return(float *out, float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float v = categorize(in[tid]);
        out[tid] = v;
        if (tid == 0) {
            printf("n=%d\n", n);
        }
    }
}
