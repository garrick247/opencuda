// Probe: complex type casting chains, integer promotion rules,
// signed/unsigned comparison edge cases, overflow-sensitive arithmetic

// Cast chain: int -> float -> double -> int
__global__ void cast_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        float f = (float)x;
        double d = (double)f;
        int back = (int)d;
        out[tid] = back + (int)(d * 0.5);
    }
}

// Signed vs unsigned comparison — explicit casts
__global__ void sign_compare(int *out, int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sa = a[tid];
        unsigned int ub = b[tid];
        // Compare signed to unsigned via explicit cast
        int result = ((unsigned int)sa < ub) ? 1 :
                     ((unsigned int)sa > ub) ? -1 : 0;
        out[tid] = result;
    }
}

// Integer promotion: small types promoted to int in arithmetic
__global__ void int_promo(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Bit operations that could cause sign issues
        unsigned int hi = v >> 16;
        unsigned int lo = v & 0xFFFF;
        unsigned int mixed = (hi << 8) | (lo >> 8);
        out[tid] = mixed ^ (v >> 1);
    }
}

// Large literal constants
__global__ void large_consts(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = (long long)tid;
        long long large = 0x7FFFFFFFFFFFFFFFLL;
        long long mask  = 0x00FF00FF00FF00FFLL;
        out[tid] = (v * 1000000000LL) ^ (large & mask);
    }
}
