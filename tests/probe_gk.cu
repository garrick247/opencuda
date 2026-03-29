// Probe: 64-bit division and modulo — these require sign-aware PTX
// div.s64 / div.u64 / rem.s64 / rem.u64

__global__ void div64_signed(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long av = a[tid];
        long long bv = b[tid];
        if (bv != 0LL) {
            out[tid] = av / bv;
        } else {
            out[tid] = 0LL;
        }
    }
}

__global__ void mod64_signed(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long av = a[tid];
        long long bv = b[tid];
        if (bv != 0LL) {
            out[tid] = av % bv;
        } else {
            out[tid] = av;
        }
    }
}

__global__ void div64_unsigned(unsigned long long *out, unsigned long long *a,
                                 unsigned long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long av = a[tid];
        unsigned long long bv = b[tid];
        if (bv != 0ULL) {
            out[tid] = av / bv;
        } else {
            out[tid] = 0ULL;
        }
    }
}

// Mixed: 64-bit value divided by 32-bit constant
__global__ void div64_by_const(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] / 1000000007LL;
    }
}
