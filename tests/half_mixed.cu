__device__ float half_clamp(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__global__ void half_mixed(float *out, half *in, float lo, float hi, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = (float)in[tid];
        out[tid] = half_clamp(v, lo, hi);
    }
}
