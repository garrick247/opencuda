// Load f32, multiply, store result as f32.
// Load second f32, compute f64 intermediate, store back as f32.
// Tests cvt paths around stores.
__global__ void nasty_mem_widen_narrow(float *a, float *b, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float x = a[tid];
        float y = x * 2.0f;
        out[tid] = y;
        float z = b[tid];
        double d = (double)z * 1.5;
        out[tid + n] = (float)d;
    }
}
