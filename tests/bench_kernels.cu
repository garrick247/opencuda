// Benchmark kernels compiled by OpenCUDA. Simple enough for fair comparison.

__global__ void bench_vadd(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] + b[gid];
}

__global__ void bench_saxpy(float *out, float alpha, float *x, float *y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = alpha * x[gid] + y[gid];
}

__global__ void bench_reduce(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
