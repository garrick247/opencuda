// Regression: float/double division requires a modifier in PTX.
// f32: div.approx.f32 (matches CUDA default fast-math /),
// f64: div.rn.f64 (IEEE round-to-nearest, rounding modifier mandatory).
// Without fix: "div.f32" / "div.f64" rejected by ptxas with
//   "Rounding modifier or '.approx' modifier required for instruction 'div'"
__global__ void float_div_test(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float q = a[tid] / b[tid];    // div.approx.f32
        float r = a[tid] / 2.0f;     // const divisor — still div.approx.f32
        out[tid * 2 + 0] = q;
        out[tid * 2 + 1] = r;
    }
}

__global__ void double_div_test(double *out, double *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double q = a[tid] / b[tid];   // div.rn.f64
        double r = 1.0 / a[tid];      // const numerator — div.rn.f64 with 0d literal
        out[tid * 2 + 0] = q;
        out[tid * 2 + 1] = r;
    }
}
