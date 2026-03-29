// Probe: signed/unsigned mixed comparisons, unsigned wraparound,
// pointer difference, size_t arithmetic, and integer edge cases.

// ------------------------------------------------------------------
// Unsigned comparison: u32 vs s32 pitfall.

__global__ void unsigned_cmp(int *out, unsigned int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int ua = a[tid];
        int sb = b[tid];
        // C: if one operand is unsigned, the signed one converts.
        // -1 as unsigned is UINT_MAX — must compare correctly.
        int r = 0;
        if (ua > (unsigned int)sb) r = 1;
        if ((int)ua < sb) r += 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Unsigned arithmetic wraparound.

__global__ void unsigned_wrap(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int a = v + 0xFFFFFFFFu;  // wraps to v - 1
        unsigned int b = v * 2u;
        unsigned int c = v >> 1;
        unsigned int d = (v - 1u) & 0xFFu;  // low byte of (v-1)
        out[tid] = a ^ b ^ c ^ d;
    }
}

// ------------------------------------------------------------------
// Long long arithmetic: 64-bit signed operations.

__global__ void ll_arith(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long av = a[tid], bv = b[tid];
        long long sum  = av + bv;
        long long diff = av - bv;
        long long prod = av * bv;
        long long shr  = av >> 3;
        out[tid] = sum + diff + prod + shr;
    }
}

// ------------------------------------------------------------------
// Unsigned long long: 64-bit unsigned.

__global__ void ull_arith(unsigned long long *out,
                          unsigned long long *a,
                          unsigned long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long av = a[tid], bv = b[tid];
        unsigned long long r = (av & bv) | (av ^ bv);
        r += av * bv;
        r >>= 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Integer min/max via ternary (no intrinsic).

__global__ void int_minmax(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        int lo = (av < bv) ? av : bv;
        int hi = (av > bv) ? av : bv;
        out[tid * 2 + 0] = lo;
        out[tid * 2 + 1] = hi;
    }
}

// ------------------------------------------------------------------
// Chained comparisons: a < b < c via separate boolean ops.

__global__ void chained_cmp(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        // a < b && b < c: b in range (a, c)
        int in_range = (av < bv) && (bv < cv);
        // a <= b <= c: non-decreasing
        int nondec = (av <= bv) && (bv <= cv);
        out[tid] = in_range + nondec * 2;
    }
}

// ------------------------------------------------------------------
// Bitfield-style packing/unpacking via shifts and masks.

__global__ void bitfield(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Pack: [7:0]=low8, [15:8]=mid8, [23:16]=hi8
        unsigned int lo  = v & 0xFFu;
        unsigned int mid = (v >> 8) & 0xFFu;
        unsigned int hi  = (v >> 16) & 0xFFu;
        // Swap lo and hi
        unsigned int packed = (lo << 16) | (mid << 8) | hi;
        out[tid] = packed;
    }
}

// ------------------------------------------------------------------
// Integer overflow check pattern (using subtraction trick).

__global__ void overflow_check(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        // Safe add check: would a+b overflow?
        // Using cast to long long
        long long sum_ll = (long long)av + (long long)bv;
        int overflows = (sum_ll > 2147483647LL) || (sum_ll < -2147483648LL);
        out[tid] = overflows ? 0 : (av + bv);
    }
}
