// Probe: __funnelshift_l/r, __byte_perm, extra atomics (atomicAnd/Or/Xor/Exch/CAS),
// __float_as_int/__int_as_float bit-cast, union type-pun, __hadd_sat/__hmul_sat,
// __hfma, pointer output multi-result, and nested array-of-struct access.

// ------------------------------------------------------------------
// __funnelshift_l / __funnelshift_r / __funnelshift_lc / __funnelshift_rc

__global__ void funnel_shift(unsigned *out_l, unsigned *out_r,
                               unsigned *a, unsigned *b, unsigned *sh, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_l[tid] = __funnelshift_l(a[tid], b[tid], sh[tid]);
        out_r[tid] = __funnelshift_r(a[tid], b[tid], sh[tid]);
    }
}

__global__ void funnel_shift_clamp(unsigned *out_lc, unsigned *out_rc,
                                     unsigned *a, unsigned *b, unsigned *sh, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_lc[tid] = __funnelshift_lc(a[tid], b[tid], sh[tid]);
        out_rc[tid] = __funnelshift_rc(a[tid], b[tid], sh[tid]);
    }
}

// ------------------------------------------------------------------
// __byte_perm: select 4 bytes from two 32-bit words using a 16-bit selector.

__global__ void byte_perm_kernel(unsigned *out, unsigned *a, unsigned *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // selector 0x4523 = bytes [5,4,2,3] from {b,a}
        out[tid] = __byte_perm(a[tid], b[tid], 0x4523);
    }
}

// ------------------------------------------------------------------
// Additional atomics: atomicAnd, atomicOr, atomicXor, atomicExch, atomicCAS.

__global__ void atomic_bitwise(int *dst_and, int *dst_or, int *dst_xor,
                                  int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicAnd(dst_and, in[gid]);
        atomicOr (dst_or,  in[gid]);
        atomicXor(dst_xor, in[gid]);
    }
}

__global__ void atomic_exch_cas(int *slot, int *old_out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int old = atomicExch(slot, in[gid]);
        old_out[gid] = old;
    }
}

__global__ void atomic_cas_kernel(int *slot, int *out, int expected, int desired, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = atomicCAS(slot, expected, desired);
    }
}

// ------------------------------------------------------------------
// __float_as_int / __int_as_float / __double_as_longlong / __longlong_as_double

__global__ void float_int_bits(int *out_bits, float *out_float,
                                  float *inf, int *ini, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   bits = __float_as_int(inf[tid]);
        float back = __int_as_float(ini[tid]);
        out_bits[tid]  = bits;
        out_float[tid] = back;
    }
}

__global__ void double_ll_bits(long long *out_bits, double *out_d,
                                  double *ind, long long *inll, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long bits = __double_as_longlong(ind[tid]);
        double    back = __longlong_as_double(inll[tid]);
        out_bits[tid] = bits;
        out_d[tid]    = back;
    }
}

// ------------------------------------------------------------------
// Union type-pun: float <-> unsigned via union.

union FloatBits {
    float    f;
    unsigned u;
};

__global__ void union_pun(unsigned *out_u, float *out_f,
                            float *inf, unsigned *inu, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union FloatBits fb;
        fb.f = inf[tid];
        out_u[tid] = fb.u;

        union FloatBits gb;
        gb.u = inu[tid];
        out_f[tid] = gb.f;
    }
}

// ------------------------------------------------------------------
// __hadd_sat / __hmul_sat (saturating half ops).

__global__ void half_sat_ops(unsigned short *out_add, unsigned short *out_mul,
                               unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half rsa = __hadd_sat(ha, hb);
        __half rsm = __hmul_sat(ha, hb);
        out_add[tid] = __half_as_ushort(rsa);
        out_mul[tid] = __half_as_ushort(rsm);
    }
}

// ------------------------------------------------------------------
// __hfma (without saturation).

__global__ void half_fma(unsigned short *out,
                          unsigned short *a, unsigned short *b,
                          unsigned short *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half hc = __ushort_as_half(c[tid]);
        __half r = __hfma(ha, hb, hc);
        out[tid] = __half_as_ushort(r);
    }
}

// ------------------------------------------------------------------
// Array of structs.

struct Point3 {
    float x, y, z;
};

__global__ void aos_distance(float *out, struct Point3 *pts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float dx = pts[tid].x;
        float dy = pts[tid].y;
        float dz = pts[tid].z;
        out[tid] = dx*dx + dy*dy + dz*dz;  // squared distance from origin
    }
}

// ------------------------------------------------------------------
// Multiple outputs via pointer params from __device__ function.

__device__ void minmax(int *lo, int *hi, int a, int b) {
    *lo = (a < b) ? a : b;
    *hi = (a < b) ? b : a;
}

__global__ void minmax_kernel(int *out_lo, int *out_hi,
                                int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lo, hi;
        minmax(&lo, &hi, a[tid], b[tid]);
        out_lo[tid] = lo;
        out_hi[tid] = hi;
    }
}
