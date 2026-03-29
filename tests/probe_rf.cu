// Probe: emitter stress — many live values, double precision mixing,
// wide multiply, large constants, and predicate-heavy code.

// ------------------------------------------------------------------
// Many simultaneous live values: tests register allocator.

__global__ void many_live(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Keep many values live simultaneously
        float a = v + 1.0f;
        float b = v + 2.0f;
        float c = v + 3.0f;
        float d = v + 4.0f;
        float e = v + 5.0f;
        float f = v + 6.0f;
        float g = v + 7.0f;
        float h = v + 8.0f;
        float i = v + 9.0f;
        float j = v + 10.0f;
        float k = v + 11.0f;
        float l = v + 12.0f;
        // Use all of them in a complex expression
        out[tid] = (a*b + c*d) * (e*f + g*h) + (i*j + k*l);
    }
}

// ------------------------------------------------------------------
// Double precision: explicit double arithmetic.

__global__ void double_arith(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double r = v * v + 2.0 * v + 1.0;  // (v+1)^2
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Mixed double and float: explicit conversions.

__global__ void mixed_precision(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float  fv  = in[tid];
        double dv  = (double)fv;
        double dr  = dv * dv + 1.0;
        float  fr  = (float)dr;
        out[tid] = fr;
    }
}

// ------------------------------------------------------------------
// 32x32 → 64 bit multiply: tests mul.wide or mul.lo.s64.

__global__ void wide_mul(long long *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long la = (long long)a[tid];
        long long lb = (long long)b[tid];
        out[tid] = la * lb;
    }
}

// ------------------------------------------------------------------
// Large constant values: near-INT32_MAX, near-INT32_MIN.

__global__ void large_const(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r = v & 0x7FFFFFFF;  // mask to non-negative
        if (r > 1000000) r = 1000000;
        if (r < -1000000) r = -1000000;
        out[tid] = r + 2147483647;  // INT32_MAX as constant
    }
}

// ------------------------------------------------------------------
// Many predicates: long chain of conditionals creating many %p regs.

__global__ void many_preds(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int score = 0;
        // Each condition uses a separate predicate
        if (v > 10)   score += 1;
        if (v > 20)   score += 2;
        if (v > 30)   score += 4;
        if (v > 40)   score += 8;
        if (v > 50)   score += 16;
        if (v > 60)   score += 32;
        if (v > 70)   score += 64;
        if (v > 80)   score += 128;
        if (v > 90)   score += 256;
        if (v > 100)  score += 512;
        if (v < -10)  score -= 1;
        if (v < -20)  score -= 2;
        if (v < -30)  score -= 4;
        if (v < -40)  score -= 8;
        out[tid] = score;
    }
}

// ------------------------------------------------------------------
// Float special values: test that 0.0f, -0.0f, 1.0f, -1.0f are encoded.

__global__ void float_specials(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float vals[4];
        vals[0] =  0.0f;
        vals[1] = -0.0f;
        vals[2] =  1.0f;
        vals[3] = -1.0f;
        out[tid] = vals[tid % 4];
    }
}
