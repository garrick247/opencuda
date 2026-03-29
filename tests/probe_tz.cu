// Probe: special integer ops — byte_perm, funnel shift, SAD, hadd,
// mulhi, mul24, and integer min/max intrinsics.

// ------------------------------------------------------------------
// __byte_perm.

__global__ void byte_perm_kernel(unsigned int *out, unsigned int *a,
                                  unsigned int *b, unsigned int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __byte_perm(a[tid], b[tid], sel[tid]);
    }
}

// ------------------------------------------------------------------
// __funnelshift_l and __funnelshift_r.

__global__ void funnel_shift(unsigned int *out, unsigned int *lo,
                               unsigned int *hi, int shift, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid * 2 + 0] = __funnelshift_l(lo[tid], hi[tid], shift);
        out[tid * 2 + 1] = __funnelshift_r(lo[tid], hi[tid], shift);
    }
}

// ------------------------------------------------------------------
// __sad (sum of absolute differences).

__global__ void sad_kernel(unsigned int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int result = 0;
        for (int i = 0; i < 4; i++) {
            result = __sad(a[tid * 4 + i], b[tid * 4 + i], result);
        }
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// __mulhi (multiply high 32 bits).

__global__ void mulhi_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __mulhi(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// __mul24 (24-bit multiply).

__global__ void mul24_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __mul24(a[tid] & 0xFFFFFF, b[tid] & 0xFFFFFF);
    }
}

// ------------------------------------------------------------------
// __hadd (halving add — no overflow).

__global__ void hadd_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __hadd(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// __usad (unsigned SAD).

__global__ void usad_kernel(unsigned int *out, unsigned int *a,
                              unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int result = 0;
        for (int i = 0; i < 4; i++) {
            result = __usad(a[tid * 4 + i], b[tid * 4 + i], result);
        }
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// min/max on int (PTX min.s32/max.s32).

__global__ void int_minmax(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = min(v, 100);
        int b = max(v, -100);
        out[tid] = a + b;
    }
}
