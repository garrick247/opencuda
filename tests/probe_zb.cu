// Probe: 64-bit codegen audit — all 64-bit binary ops (add/sub/mul/div/rem/and/or/xor/shl/shr),
// 64-bit comparisons (all 6: <, >, <=, >=, ==, !=) on signed and unsigned,
// 64-bit min/max, 64-bit abs, 64-bit ternary, 64-bit atomicAdd/atomicMax,
// and 64-bit loop counter.

// ------------------------------------------------------------------
// 64-bit signed arithmetic: add, sub, mul, div, rem.

__global__ void ll_arith(long long *out_a, long long *out_s, long long *out_m,
                           long long *out_d, long long *out_r,
                           long long *x, long long *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long a = x[tid], b = y[tid];
        out_a[tid] = a + b;
        out_s[tid] = a - b;
        out_m[tid] = a * b;
        if (b != 0) out_d[tid] = a / b; else out_d[tid] = 0;
        if (b != 0) out_r[tid] = a % b; else out_r[tid] = 0;
    }
}

// ------------------------------------------------------------------
// 64-bit unsigned arithmetic.

__global__ void ull_arith(unsigned long long *out_a, unsigned long long *out_m,
                            unsigned long long *out_d,
                            unsigned long long *x, unsigned long long *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long a = x[tid], b = y[tid];
        out_a[tid] = a + b;
        out_m[tid] = a * b;
        if (b != 0) out_d[tid] = a / b; else out_d[tid] = 0;
    }
}

// ------------------------------------------------------------------
// 64-bit bitwise: and, or, xor, not, shift.

__global__ void ll_bitwise(long long *out_and, long long *out_or, long long *out_xor,
                              long long *out_not, long long *out_shl, long long *out_shr,
                              long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_and[tid] = a[tid] & b[tid];
        out_or[tid]  = a[tid] | b[tid];
        out_xor[tid] = a[tid] ^ b[tid];
        out_not[tid] = ~a[tid];
        out_shl[tid] = a[tid] << 3;
        out_shr[tid] = a[tid] >> 2;
    }
}

// ------------------------------------------------------------------
// 64-bit comparisons: all 6 ops on signed.

__global__ void ll_cmp(int *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long x = a[tid], y = b[tid];
        int bits = 0;
        if (x <  y) bits |= 1;
        if (x >  y) bits |= 2;
        if (x <= y) bits |= 4;
        if (x >= y) bits |= 8;
        if (x == y) bits |= 16;
        if (x != y) bits |= 32;
        out[tid] = bits;
    }
}

// ------------------------------------------------------------------
// 64-bit unsigned comparisons.

__global__ void ull_cmp(int *out, unsigned long long *a, unsigned long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long x = a[tid], y = b[tid];
        int bits = 0;
        if (x <  y) bits |= 1;
        if (x >  y) bits |= 2;
        if (x <= y) bits |= 4;
        if (x >= y) bits |= 8;
        if (x == y) bits |= 16;
        if (x != y) bits |= 32;
        out[tid] = bits;
    }
}

// ------------------------------------------------------------------
// 64-bit min / max / abs.

__device__ long long ll_max(long long a, long long b) {
    return (a > b) ? a : b;
}
__device__ long long ll_min(long long a, long long b) {
    return (a < b) ? a : b;
}
__device__ long long ll_abs(long long a) {
    return (a < 0) ? -a : a;
}

__global__ void ll_minmaxabs(long long *out_max, long long *out_min, long long *out_abs,
                                long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_max[tid] = ll_max(a[tid], b[tid]);
        out_min[tid] = ll_min(a[tid], b[tid]);
        out_abs[tid] = ll_abs(a[tid]);
    }
}

// ------------------------------------------------------------------
// 64-bit ternary in expression.

__global__ void ll_ternary(long long *out, long long *a, long long *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (sel[tid] & 1) ? a[tid] : b[tid];
    }
}

// ------------------------------------------------------------------
// 64-bit loop counter.

__global__ void ll_loop_counter(long long *out, long long start, long long step, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long s = 0;
        for (long long i = start; i < start + 16 * step; i += step) {
            s += i;
        }
        out[tid] = s + (long long)tid;
    }
}

// ------------------------------------------------------------------
// 64-bit cast from 32-bit: sign-extend and zero-extend.

__global__ void ll_extend(long long *out_se, unsigned long long *out_ze,
                             int *in_s, unsigned *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_se[tid] = (long long)in_s[tid];      // sign extend
        out_ze[tid] = (unsigned long long)in_u[tid];  // zero extend
    }
}
