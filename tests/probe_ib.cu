// Probe: integer/float mixed arithmetic with explicit casts,
// unsigned comparisons, signed/unsigned mixing,
// bitwise operations on float bits

__global__ void mixed_arith(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   ival = in[tid];
        float fval = (float)ival;
        float scaled = fval * 0.001f;
        int   back   = (int)(scaled * 1000.0f);
        unsigned int ubits = (unsigned int)back;
        unsigned int masked = ubits & 0xFFFF;
        out[tid] = (float)masked + scaled;
    }
}

// Unsigned comparison edge cases
__global__ void unsigned_compare(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int x = a[tid];
        unsigned int y = b[tid];
        // unsigned division and comparison
        unsigned int q = (y != 0u) ? (x / y) : 0u;
        unsigned int r = (y != 0u) ? (x % y) : x;
        out[tid] = (q > r) ? q - r : r - q;
    }
}

// Float bit manipulation via reinterpret
__global__ void float_bits(unsigned int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = in[tid];
        unsigned int bits = *((unsigned int *)&f);
        unsigned int sign = bits >> 31;
        unsigned int exp  = (bits >> 23) & 0xFF;
        unsigned int mant = bits & 0x7FFFFF;
        out[tid] = sign * 1000000u + exp * 1000u + (mant >> 13);
    }
}

// Integer overflow patterns (well-defined unsigned)
__global__ void uint_overflow(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Multiply-high approximation
        unsigned long long wide = (unsigned long long)v * (unsigned long long)v;
        unsigned int hi = (unsigned int)(wide >> 32);
        unsigned int lo = (unsigned int)(wide & 0xFFFFFFFF);
        out[tid] = hi ^ lo;
    }
}
