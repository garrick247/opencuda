__global__ void saxpy(float *out, float alpha, float *x, float *y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = alpha * x[gid] + y[gid];
}