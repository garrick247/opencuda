// A __device__ function that takes 4 mixed-type args (int, float, float, int)
// and returns a float. Kernel calls it with values loaded from memory.
__device__ float mix_compute(int a, float b, float c, int d) {
    return (float)a * b + c * (float)d;
}

__global__ void nasty_mem_call_args(int *ia, float *fb, float *fc, int *id, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int a = ia[tid];
        float b = fb[tid];
        float c = fc[tid];
        int d = id[tid];
        out[tid] = mix_compute(a, b, c, d);
    }
}
