__global__ void half_edges(float *out, half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = (float)in[tid];
        if (f > 65504.0f) f = 65504.0f;
        if (f < -65504.0f) f = -65504.0f;
        out[tid] = f;
    }
}
