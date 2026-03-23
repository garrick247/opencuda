// Liveness test: chain of short-lived values — linear scan should reuse registers.
// Optimal: 2-3 f32 registers (each value dies before the next is born).
// Test asserts <= 4 f32 registers declared.
__global__ void chain_reuse(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid];
        float b = a * 2.0f;
        float c = b + 1.0f;
        float d = c * c;
        float e = d - 0.5f;
        float f2 = e + e;
        out[tid] = f2;
    }
}
