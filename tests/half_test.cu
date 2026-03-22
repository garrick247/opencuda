__global__ void half_test(float *out, half *a) {
    int tid = threadIdx.x;
    half v = a[tid];
    out[tid] = (float)v;
}
