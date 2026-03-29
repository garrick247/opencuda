// Probe: half-precision unary ops (__hneg, __habs, __hrcp, __hrsqrt, __hexp,
// __hlog, __hsqrt), half comparison (__hgt, __hlt, __heq, __hne),
// __hfma_sat, and half-precision FP reduction via shfl+hadd.

// ------------------------------------------------------------------
// __hneg and __habs.

__global__ void half_neg_abs(unsigned short *out_neg, unsigned short *out_abs,
                               unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        __half neg = __hneg(v);
        __half abs = __habs(v);
        out_neg[tid] = __half_as_ushort(neg);
        out_abs[tid] = __half_as_ushort(abs);
    }
}

// ------------------------------------------------------------------
// __hrcp and __hrsqrt.

__global__ void half_rcp_rsqrt(unsigned short *out_rcp, unsigned short *out_rsqrt,
                                  unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        __half rcp   = __hrcp(v);
        __half rsqrt = __hrsqrt(v);
        out_rcp[tid]   = __half_as_ushort(rcp);
        out_rsqrt[tid] = __half_as_ushort(rsqrt);
    }
}

// ------------------------------------------------------------------
// __hexp and __hlog.

__global__ void half_exp_log(unsigned short *out_exp, unsigned short *out_log,
                               unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        __half ex = __hexp(v);
        __half lg = __hlog(v);
        out_exp[tid] = __half_as_ushort(ex);
        out_log[tid] = __half_as_ushort(lg);
    }
}

// ------------------------------------------------------------------
// __hsub and __hdiv.

__global__ void half_sub_div(unsigned short *out_sub, unsigned short *out_div,
                               unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        out_sub[tid] = __half_as_ushort(__hsub(ha, hb));
        out_div[tid] = __half_as_ushort(__hdiv(ha, hb));
    }
}

// ------------------------------------------------------------------
// Half comparisons: __hgt, __hlt, __heq, __hne, __hge, __hle.

__global__ void half_compare(int *out, unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        int gt = __hgt(ha, hb);
        int lt = __hlt(ha, hb);
        int eq = __heq(ha, hb);
        int ne = __hne(ha, hb);
        // Encode as 4 bits
        out[tid] = (gt << 3) | (lt << 2) | (eq << 1) | ne;
    }
}

// ------------------------------------------------------------------
// __hfma_sat (fused multiply-add with saturation to [0,1]).

__global__ void half_fma_sat(unsigned short *out,
                               unsigned short *a, unsigned short *b,
                               unsigned short *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half hc = __ushort_as_half(c[tid]);
        __half r = __hfma_sat(ha, hb, hc);
        out[tid] = __half_as_ushort(r);
    }
}

// ------------------------------------------------------------------
// Half-precision warp reduce via shfl + hadd.
// Each warp computes the sum of 32 half values.

__global__ void half_warp_sum(float *out, unsigned short *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Load as half, convert to float for shfl (shfl on half would need b16 shfl)
    float v = (gid < n) ? __half2float(__ushort_as_half(in[gid])) : 0.0f;
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if ((threadIdx.x & 31) == 0 && gid < n) {
        out[gid / 32] = v;
    }
}

// ------------------------------------------------------------------
// __hisnan and __hisinf.

__global__ void half_special(int *out_nan, int *out_inf,
                               unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        out_nan[tid] = __hisnan(v);
        out_inf[tid] = __hisinf(v);
    }
}
