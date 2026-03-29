// Regression: ternary with mixed int/float arms.
// result_ty is float (float wins promotion), but the false-arm is an int Value.
// Without fix: emits mov.f32 %f0, %r5 (type mismatch — ptxas rejects it).
// With fix: _coerce_to_float emits cvt.rn.f32.s32 before the mov.
__global__ void ternary_mixed_test(float *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        // int-arm ternary: the false branch is an int variable
        float a = (i > 0) ? 1.0f : i;   // int arm needs cvt to float
        // both-arms ternary with different types
        float b = (i < n) ? (float)i : 0.0f;
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}
