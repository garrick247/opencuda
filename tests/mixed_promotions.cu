__global__ void mixed_promotions(float *out, int *a, float *b, half *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float ai = (float)a[tid];
        float bf = b[tid];
        float cv = (float)c[tid];
        out[tid] = ai * bf + cv;
    }
}
