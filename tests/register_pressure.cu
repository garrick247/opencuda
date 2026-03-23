__global__ void register_pressure(float *out, float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float a = in[tid * 8 + 0];
        float b = in[tid * 8 + 1];
        float c = in[tid * 8 + 2];
        float d = in[tid * 8 + 3];
        float e = in[tid * 8 + 4];
        float f = in[tid * 8 + 5];
        float g = in[tid * 8 + 6];
        float h = in[tid * 8 + 7];
        float sum  = a + b + c + d + e + f + g + h;
        float prod = a * b + c * d + e * f + g * h;
        float cross = (a + c + e + g) * (b + d + f + h);
        out[tid] = sum + prod + cross;
    }
}
