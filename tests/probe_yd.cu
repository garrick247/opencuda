// Probe: __fsqrt_rn/__fsqrt_rd, __frsqrt_rn, __frcp_rn/__frcp_rd,
// __fmaf_rn (float fma), __fmaf_ieee_rn, fabs/fabsf/fmaxf/fminf,
// __saturatef, floor/ceil/round/trunc (float), ldexp, frexp-style,
// double math: sqrt/floor/ceil/fabs, int64 ops (__ll2float_rn, __float2ll_rn),
// unsigned 64-bit to float, and __hiloint2double/__double2hiint/__double2loint.

// ------------------------------------------------------------------
// __fsqrt_rn, __fsqrt_rd.

__global__ void fsqrt_modes(float *out_rn, float *out_rd, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_rn[tid] = __fsqrt_rn(in[tid]);
        out_rd[tid] = __fsqrt_rd(in[tid]);
    }
}

// ------------------------------------------------------------------
// __frsqrt_rn.

__global__ void frsqrt_rn(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __frsqrt_rn(in[tid]);
}

// ------------------------------------------------------------------
// __frcp_rn / __frcp_rd.

__global__ void frcp_modes(float *out_rn, float *out_rd, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_rn[tid] = __frcp_rn(in[tid]);
        out_rd[tid] = __frcp_rd(in[tid]);
    }
}

// ------------------------------------------------------------------
// __fmaf_rn (float fused multiply-add, round-nearest).

__global__ void fmaf_kernel(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __fmaf_rn(a[tid], b[tid], c[tid]);
}

// ------------------------------------------------------------------
// fabs / fabsf / fmaxf / fminf (standard math aliases).

__global__ void std_math_float(float *out_abs, float *out_max, float *out_min,
                                  float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_abs[tid] = fabsf(a[tid]);
        out_max[tid] = fmaxf(a[tid], b[tid]);
        out_min[tid] = fminf(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// __saturatef: clamps to [0.0, 1.0].

__global__ void saturatef_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __saturatef(in[tid]);
}

// ------------------------------------------------------------------
// floorf / ceilf / roundf / truncf.

__global__ void float_round_modes(float *out_fl, float *out_ce,
                                    float *out_ro, float *out_tr,
                                    float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_fl[tid] = floorf(in[tid]);
        out_ce[tid] = ceilf(in[tid]);
        out_ro[tid] = roundf(in[tid]);
        out_tr[tid] = truncf(in[tid]);
    }
}

// ------------------------------------------------------------------
// Double math: sqrt, floor, ceil, fabs.

__global__ void double_math(double *out_sqrt, double *out_fl,
                               double *out_abs, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_sqrt[tid] = sqrt(in[tid]);
        out_fl[tid]   = floor(in[tid]);
        out_abs[tid]  = fabs(in[tid]);
    }
}

// ------------------------------------------------------------------
// __ll2float_rn / __float2ll_rn: int64 ↔ float.

__global__ void ll_float_cvt(float *out_f, long long *out_ll,
                               long long *in_ll, float *in_f, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_f[tid]  = __ll2float_rn(in_ll[tid]);
        out_ll[tid] = __float2ll_rn(in_f[tid]);
    }
}

// ------------------------------------------------------------------
// unsigned 64-bit to float: __ull2float_rn.

__global__ void ull2float_kernel(float *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __ull2float_rn(in[tid]);
}

// ------------------------------------------------------------------
// __hiloint2double / __double2hiint / __double2loint.

__global__ void double_parts(double *out_d, int *out_hi, int *out_lo,
                               int *hi_in, int *lo_in, double *d_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Compose double from hi/lo 32-bit words
        double d = __hiloint2double(hi_in[tid], lo_in[tid]);
        out_d[tid]  = d;
        // Decompose
        out_hi[tid] = __double2hiint(d_in[tid]);
        out_lo[tid] = __double2loint(d_in[tid]);
    }
}
