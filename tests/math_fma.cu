// fmaf(a,b,c) = fma.rn.f32 dest, a, b, c (single fused multiply-add)
// Without fix: dest register uninitialized (no PTX emission, wrong INT32 return type)
__global__ void math_fma_test(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = a[tid];
        float y = b[tid];
        float z = c[tid];
        out[tid] = fmaf(x, y, z);
    }
}
