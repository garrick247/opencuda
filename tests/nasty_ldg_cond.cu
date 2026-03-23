// Nasty: __ldg + pointer arithmetic inside a conditional inside a loop.
// Tests that ld.global.nc is emitted consistently when the load is guarded.
__global__ void conditional_ldg(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* out, int n, int mode) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float va = __ldg(a + i);
        if (mode == 0) {
            float vb = __ldg(b + i);
            sum = sum + va * vb;
        } else {
            sum = sum + va;
        }
    }
    out[tid] = sum;
}
