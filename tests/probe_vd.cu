// Probe: double precision math, integer division patterns,
// NaN-sensitive float comparisons, and large struct passing.

// ------------------------------------------------------------------
// Double precision: basic ops, transcendentals, comparison.

__global__ void double_math(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double a = sqrt(v > 0.0 ? v : -v);
        double b = sin(v) + cos(v);
        double c = exp(v < 10.0 ? v : 10.0);
        double d = log(v > 0.0 ? v : 1.0);
        out[tid] = a + b + c + d;
    }
}

// ------------------------------------------------------------------
// Double precision loop accumulator.

__global__ void double_accum(double *out, double *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        double acc = 0.0;
        for (int i = 0; i < k; i++) {
            double v = in[tid * k + i];
            acc += v * v;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Integer division / modulo patterns.

__global__ void div_mod_patterns(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v / 2;        // div by power of 2
        int b = v % 2;        // mod by power of 2
        int c = v / 3;        // div by odd constant
        int d = v % 7;        // mod by constant
        int e = v / 16;       // div by power of 2
        int f = v % 256;      // mod by power of 2
        out[tid] = a + b + c + d + e + f;
    }
}

// ------------------------------------------------------------------
// NaN-sensitive comparisons (ordered vs unordered).

__global__ void nan_compare(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // IEEE: any comparison with NaN is false (ordered comparisons)
        int r = 0;
        if (v > 0.0f)  r += 1;   // false for NaN
        if (v < 0.0f)  r += 2;   // false for NaN
        if (v == 0.0f) r += 4;   // false for NaN
        if (v != v)    r += 8;   // true only for NaN (NaN != NaN)
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float with explicit infinity / special value handling.

__global__ void float_special(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Clamp to reasonable range
        float clamped = v > 1e30f ? 1e30f : (v < -1e30f ? -1e30f : v);
        // Safe reciprocal
        float recip = (v == 0.0f) ? 0.0f : 1.0f / v;
        out[tid] = clamped * recip;
    }
}

// ------------------------------------------------------------------
// Large struct (8 floats + 4 ints).

struct BigParam {
    float f0, f1, f2, f3;
    float f4, f5, f6, f7;
    int   i0, i1, i2, i3;
};

__device__ float big_compute(struct BigParam p) {
    float fsum = p.f0 + p.f1 + p.f2 + p.f3
               + p.f4 + p.f5 + p.f6 + p.f7;
    int isum = p.i0 + p.i1 + p.i2 + p.i3;
    return fsum + (float)isum;
}

__global__ void large_struct_call(float *out, float *fin, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct BigParam p;
        p.f0 = fin[tid * 8 + 0]; p.f1 = fin[tid * 8 + 1];
        p.f2 = fin[tid * 8 + 2]; p.f3 = fin[tid * 8 + 3];
        p.f4 = fin[tid * 8 + 4]; p.f5 = fin[tid * 8 + 5];
        p.f6 = fin[tid * 8 + 6]; p.f7 = fin[tid * 8 + 7];
        p.i0 = iin[tid * 4 + 0]; p.i1 = iin[tid * 4 + 1];
        p.i2 = iin[tid * 4 + 2]; p.i3 = iin[tid * 4 + 3];
        out[tid] = big_compute(p);
    }
}

// ------------------------------------------------------------------
// Mixed double and float in same kernel (implicit promotion).

__global__ void mixed_fp(double *out, float *fin, double *din, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fv = fin[tid];
        double dv = din[tid];
        // Promote float to double for arithmetic
        double result = (double)fv * dv + sqrt(dv * dv + (double)(fv * fv));
        out[tid] = result;
    }
}
