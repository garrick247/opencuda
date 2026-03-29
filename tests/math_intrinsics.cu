// Regression: math intrinsics (sqrtf, fminf, fmaxf, fabsf, etc.) must emit
// correct PTX instructions, not be silently dropped.
// Without the fix: CallInst with no PTX emission → uninitialized dest register.
__global__ void math_intrinsics_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        float sq = sqrtf(val);                     // sqrt.approx.f32
        float lo = fminf(val, 1.0f);               // min.f32
        float hi = fmaxf(val, 0.0f);               // max.f32
        float ab = fabsf(val - 0.5f);              // abs.f32
        out[tid] = sq + lo + hi + ab;
    }
}

__global__ void int_minmax_test(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lo = min(a[tid], b[tid]);              // min.s32
        int hi = max(a[tid], b[tid]);              // max.s32
        int ab = abs(a[tid]);                      // abs.s32
        out[tid] = lo + hi + ab;
    }
}
