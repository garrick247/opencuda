// Probe: bit intrinsics (__popc, __clz, __ctz, __brev, __ffs),
// math intrinsics (__fma, fmaf, __fdiv, sqrtf, rsqrtf, fabsf),
// integer min/max, isnan/isinf, and __float2int_rn/__int2float_rn.

// ------------------------------------------------------------------
// __popc / __popcll: population count.

__global__ void popc_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __popc(in[tid]);
    }
}

__global__ void popcll_kernel(long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __popcll(in[tid]);
    }
}

// ------------------------------------------------------------------
// __clz: count leading zeros.

__global__ void clz_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __clz(in[tid]);
    }
}

// ------------------------------------------------------------------
// __ffs: find first set bit (1-based, 0 if none).

__global__ void ffs_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ffs(in[tid]);
    }
}

// ------------------------------------------------------------------
// __brev: bit-reverse.

__global__ void brev_kernel(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __brev(in[tid]);
    }
}

// ------------------------------------------------------------------
// fmaf: fused multiply-add (float).

__global__ void fma_kernel(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fmaf(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// sqrtf / rsqrtf.

__global__ void sqrt_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = sqrtf(v) + rsqrtf(v + 1.0f);
    }
}

// ------------------------------------------------------------------
// fabsf / fminf / fmaxf.

__global__ void abs_minmax(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = fabsf(a[tid]);
        float bv = fabsf(b[tid]);
        out[tid] = fminf(av, bv) + fmaxf(av, bv);
    }
}

// ------------------------------------------------------------------
// min / max for integers.

__global__ void int_minmax(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[2*tid]   = min(a[tid], b[tid]);
        out[2*tid+1] = max(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// isnan / isinf.

__global__ void nan_inf_check(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int r = 0;
        if (isnan(v))  r |= 1;
        if (isinf(v))  r |= 2;
        if (isfinite(v)) r |= 4;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// __float2int_rn / __int2float_rn.

__global__ void cvt_rn(int *iout, float *fout, float *fin, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        iout[tid] = __float2int_rn(fin[tid]);
        fout[tid] = __int2float_rn(iin[tid]);
    }
}
