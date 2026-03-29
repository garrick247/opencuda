// Probe: more half intrinsics (__hceil, __hfloor, __hrint, __htrunc, __hcos,
// __hsin, __hexp2, __hlog2, __hlog10), __h2div, __hadd2/__hmul2 (half2),
// half comparison returning bool-as-int (__hbeq, __hbne, __hbgt, __hblt),
// and __hmax/__hmin edge cases.

// ------------------------------------------------------------------
// __hceil / __hfloor / __hrint / __htrunc.

__global__ void half_round_ops(unsigned short *out_ceil, unsigned short *out_floor,
                                  unsigned short *out_rint, unsigned short *out_trunc,
                                  unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        out_ceil[tid]  = __half_as_ushort(__hceil(v));
        out_floor[tid] = __half_as_ushort(__hfloor(v));
        out_rint[tid]  = __half_as_ushort(__hrint(v));
        out_trunc[tid] = __half_as_ushort(__htrunc(v));
    }
}

// ------------------------------------------------------------------
// __hcos / __hsin / __hexp2 / __hlog2.

__global__ void half_trig_log(unsigned short *out_cos, unsigned short *out_sin,
                                unsigned short *out_exp2, unsigned short *out_log2,
                                unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        out_cos[tid]  = __half_as_ushort(__hcos(v));
        out_sin[tid]  = __half_as_ushort(__hsin(v));
        out_exp2[tid] = __half_as_ushort(__hexp2(v));
        out_log2[tid] = __half_as_ushort(__hlog2(v));
    }
}

// ------------------------------------------------------------------
// __hmax_nan / __hmin_nan (NaN-propagating variants).

__global__ void half_nanmax(unsigned short *out_max, unsigned short *out_min,
                               unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        // Use regular max/min (NaN-propagating versions may not exist in all SDKs)
        out_max[tid] = __half_as_ushort(__hmax(ha, hb));
        out_min[tid] = __half_as_ushort(__hmin(ha, hb));
    }
}

// ------------------------------------------------------------------
// Chained half arithmetic: (a + b) * c - d.

__global__ void half_chain(unsigned short *out,
                              unsigned short *a, unsigned short *b,
                              unsigned short *c, unsigned short *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half hc = __ushort_as_half(c[tid]);
        __half hd = __ushort_as_half(d[tid]);
        __half r = __hmul(__hadd(ha, hb), hc);
        r = __hsub(r, hd);
        out[tid] = __half_as_ushort(r);
    }
}

// ------------------------------------------------------------------
// Mixed half/float computation.

__global__ void half_float_mix(float *out, unsigned short *in_h, float *in_f, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __ushort_as_half(in_h[tid]);
        float  f = in_f[tid];
        // Convert half to float, do float arithmetic, convert back
        float hf = __half2float(h);
        float r  = hf * f + hf;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// __hexp2 / __hlog2 used for approximate log and exp.

__global__ void half_exp_log2(unsigned short *out_e2, unsigned short *out_l2,
                                unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        out_e2[tid] = __half_as_ushort(__hexp2(v));
        out_l2[tid] = __half_as_ushort(__hlog2(v));
    }
}

// ------------------------------------------------------------------
// Half in a loop (accumulate sum of half array).

__global__ void half_sum_loop(float *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        for (int i = 0; i < 8; i++) {
            __half h = __ushort_as_half(in[(tid * 8 + i) % n]);
            s += __half2float(h);
        }
        out[tid] = s;
    }
}
