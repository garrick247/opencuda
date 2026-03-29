// Probe: float division (approx), float comparisons with NaN-like edge cases,
// float-to-int truncation semantics, double precision accumulation,
// mixed float/double expressions

// Float division: a/b and reciprocal-like patterns
__global__ void float_div_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid];
        float b = in[n - 1 - tid];
        float q = a / b;           // div.approx.f32
        float r = 1.0f / b;        // reciprocal
        out[tid * 2]     = q;
        out[tid * 2 + 1] = r;
    }
}

// Float sqrt via multiply-accumulate pattern
__global__ void float_fma(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // a*b + c — should emit mul.f32 then add.f32 (no FMA intrinsic in basic CUDA C)
        float result = a[tid] * b[tid] + c[tid];
        out[tid] = result;
    }
}

// Float accumulator in loop — no unrolling (trip count is runtime)
__global__ void float_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Double precision: simple arithmetic
__global__ void double_arith(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double a = in[tid];
        double b = 3.141592653589793;   // pi as double
        double c = a * b;
        out[tid] = c;
    }
}

// Float constant expressions: all should fold at parse time
__global__ void float_const_div(float *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float a = 10.0f / 4.0f;    // 2.5f
        float b = 7.0f / 2.0f;     // 3.5f
        float c = a + b;            // 6.0f
        out[0] = c;
        out[1] = a;
        out[2] = b;
    }
}
