// fmodf(x,y) has no direct PTX opcode — emitted as:
//   div.approx.f32 q, x, y
//   cvt.rzi.f32.f32 qt, q      (truncate toward zero)
//   mul.f32 qy, qt, y
//   sub.f32 dest, x, qy
// Without fix: dest register uninitialized (no PTX emission)
__global__ void math_fmod_test(float *out, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fmodf(x[tid], y[tid]);
    }
}
