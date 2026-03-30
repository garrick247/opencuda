// Probe: 64-bit codegen stress — long long arithmetic chain, unsigned long long
// bitwise ops, double-precision FMA (__fma_rn), double atomicAdd, 64-bit
// pointer arithmetic, large constant expressions, __brevll/__popcll/__clzll,
// __ffsll (find first set bit), sinf/cosf/expf/logf/log2f/exp2f/powf,
// and atan2f/fmodf/copysignf.

// ------------------------------------------------------------------
// Long long arithmetic chain.

__global__ void ll_chain(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long x = a[tid];
        long long y = b[tid];
        long long r = x * y + x - y;
        r = r * r + x;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Unsigned long long bitwise operations.

__global__ void ull_bitwise(unsigned long long *out, unsigned long long *a,
                               unsigned long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long x = a[tid], y = b[tid];
        unsigned long long r = (x & y) | (x ^ y);
        r = ~r;
        out[tid] = r >> 3;
    }
}

// ------------------------------------------------------------------
// fmaf (float FMA) — compiler may already fold a*b+c but this tests explicit call.

__global__ void fmaf_explicit(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fmaf(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// Double FMA: fma(double, double, double).

__global__ void dfma(double *out, double *a, double *b, double *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fma(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// __brevll (64-bit bit reverse).

__global__ void brevll_test(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __brevll(in[tid]);
}

// ------------------------------------------------------------------
// __ffsll (find first set bit in long long — returns 1-indexed position).

__global__ void ffsll_test(int *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __ffsll(in[tid]);
}

// ------------------------------------------------------------------
// __ffs (find first set bit in int).

__global__ void ffs_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __ffs(in[tid]);
}

// ------------------------------------------------------------------
// sinf / cosf / expf / logf / log2f / exp2f.

__global__ void float_math(float *out_sin, float *out_cos,
                              float *out_exp, float *out_log,
                              float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out_sin[tid] = sinf(v);
        out_cos[tid] = cosf(v);
        out_exp[tid] = expf(v);
        out_log[tid] = logf(v);
    }
}

__global__ void float_math2(float *out_log2, float *out_exp2, float *out_pow,
                               float *in_a, float *in_b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_log2[tid] = log2f(in_a[tid]);
        out_exp2[tid] = exp2f(in_a[tid]);
        out_pow[tid]  = powf(in_a[tid], in_b[tid]);
    }
}

// ------------------------------------------------------------------
// atan2f / fmodf / copysignf.

__global__ void float_math3(float *out_atan2, float *out_fmod, float *out_csign,
                               float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_atan2[tid]  = atan2f(a[tid], b[tid]);
        out_fmod[tid]   = fmodf(a[tid], b[tid]);
        out_csign[tid]  = copysignf(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Large constant expression in initializer.

__global__ void large_const(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long big1 = 1000000000LL * 1000LL;  // 10^12
        long long big2 = 0x7FFFFFFFFFFFFFFFLL;     // LLONG_MAX
        out[tid * 2    ] = big1 + (long long)tid;
        out[tid * 2 + 1] = big2 - (long long)tid;
    }
}

// ------------------------------------------------------------------
// 64-bit pointer arithmetic with struct.

struct BigData { double v[4]; };

__global__ void ptr64_struct(double *out, struct BigData *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct BigData *p = in + tid;
        double s = p->v[0] + p->v[1] + p->v[2] + p->v[3];
        out[tid] = s;
    }
}
