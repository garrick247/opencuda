// Probe: remaining intrinsic gaps — __double2float_rn/rz, atomicInc/atomicDec/atomicSub,
// __hmax/__hmin/__hfmin/__hfmax, __hadd_rn, __hgt_s/__hlt_s (signed half cmp),
// __hiloint2double/__double2hiint/__double2loint, __mul64hi/__umul64hi,
// __rhadd/__uhadd, and __float2half_rd/__float2half_ru.

// ------------------------------------------------------------------
// __double2float_rn / __double2float_rz / __double2float_rd.

__global__ void d2f_modes(float *out_rn, float *out_rz,
                            double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_rn[tid] = __double2float_rn(in[tid]);
        out_rz[tid] = __double2float_rz(in[tid]);
    }
}

// ------------------------------------------------------------------
// __float2double (widen float → double).

__global__ void f2d_widen(double *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __float2double_rn(in[tid]);
}

// ------------------------------------------------------------------
// atomicSub.

__global__ void atomic_sub(int *acc, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicSub(acc, in[gid]);
}

// ------------------------------------------------------------------
// atomicInc / atomicDec.

__global__ void atomic_inc_dec(unsigned *inc_cnt, unsigned *dec_cnt,
                                  unsigned modulo, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicInc(inc_cnt, modulo);
        atomicDec(dec_cnt, modulo);
    }
}

// ------------------------------------------------------------------
// Half max/min: __hmax / __hmin.

__global__ void half_maxmin(unsigned short *out_max, unsigned short *out_min,
                               unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        out_max[tid] = __half_as_ushort(__hmax(ha, hb));
        out_min[tid] = __half_as_ushort(__hmin(ha, hb));
    }
}

// ------------------------------------------------------------------
// __hadd_rn (round-nearest half add — same as __hadd).

__global__ void hadd_rn(unsigned short *out, unsigned short *a,
                          unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        out[tid] = __half_as_ushort(__hadd_rn(ha, hb));
    }
}

// ------------------------------------------------------------------
// __float2half_rd / __float2half_ru.

__global__ void f2h_round_modes(unsigned short *out_rd, unsigned short *out_ru,
                                   float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half hd = __float2half_rd(in[tid]);
        __half hu = __float2half_ru(in[tid]);
        out_rd[tid] = __half_as_ushort(hd);
        out_ru[tid] = __half_as_ushort(hu);
    }
}

// ------------------------------------------------------------------
// __hiloint2double / __double2hiint / __double2loint.

__global__ void hiloint_double(double *out_d, int *out_hi, int *out_lo,
                                  int *hi_in, int *lo_in, double *d_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double d = __hiloint2double(hi_in[tid], lo_in[tid]);
        out_d[tid]  = d;
        out_hi[tid] = __double2hiint(d_in[tid]);
        out_lo[tid] = __double2loint(d_in[tid]);
    }
}

// ------------------------------------------------------------------
// __mul64hi / __umul64hi.

__global__ void mul64hi_test(long long *out_hi, unsigned long long *out_uhi,
                               long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_hi[tid]  = __mul64hi(a[tid], b[tid]);
        out_uhi[tid] = __umul64hi((unsigned long long)a[tid],
                                   (unsigned long long)b[tid]);
    }
}

// ------------------------------------------------------------------
// __rhadd / __uhadd (rounding average).

__global__ void rhadd_test(int *out_r, unsigned *out_u,
                              int *a, int *b, unsigned *ua, unsigned *ub, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_r[tid] = __rhadd(a[tid], b[tid]);
        out_u[tid] = __uhadd(ua[tid], ub[tid]);
    }
}
