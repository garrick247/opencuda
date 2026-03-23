__device__ int sign_and_magnitude(int x) {
    if (x == 0) return 0;
    if (x > 0) {
        if (x > 100) return 2;
        return 1;
    } else {
        if (x < -100) return -2;
        return -1;
    }
}

__global__ void nested_returns(int *out, int *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        out[tid] = sign_and_magnitude(in[tid]);
    }
}
