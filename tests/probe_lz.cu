// Probe: integer overflow / wraparound patterns, signed vs unsigned arithmetic,
// complex integer promotions, bitwise patterns with shifts

// Unsigned wraparound (valid in C)
__global__ void uint_wraparound(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int x = (unsigned int)tid;
        unsigned int wrapped = x - 1u;  // wraps for tid=0
        out[tid] = wrapped;
    }
}

// Mixed signed/unsigned comparison
__global__ void signed_unsigned_cmp(int *out, unsigned int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int ua = a[tid];
        int sb = b[tid];
        // Comparison: promote sb to unsigned — potentially surprising
        out[tid] = (ua > (unsigned int)sb) ? 1 : 0;
    }
}

// Bit-extract: extract bits [hi:lo] from value
__device__ int extract_bits(int val, int lo, int hi) {
    int mask = ((1 << (hi - lo + 1)) - 1) << lo;
    return (val & mask) >> lo;
}

__global__ void bit_extract(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Extract bits 4..7 (nibble)
        out[tid] = extract_bits(in[tid], 4, 7);
    }
}

// Shift-based multiply/divide by power of 2
__global__ void shift_arith(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int mul8 = v << 3;      // v * 8
        int div4 = v >> 2;      // v / 4 (arithmetic shift)
        int mod16 = v & 15;     // v % 16 (for non-negative v)
        out[tid] = mul8 + div4 + mod16;
    }
}

// Bitwise accumulator
__global__ void bitwise_accum(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int xor_acc = 0u;
        unsigned int and_acc = ~0u;
        unsigned int or_acc = 0u;
        for (int i = 0; i < n; i++) {
            xor_acc ^= in[i];
            and_acc &= in[i];
            or_acc  |= in[i];
        }
        out[tid * 3]     = xor_acc;
        out[tid * 3 + 1] = and_acc;
        out[tid * 3 + 2] = or_acc;
    }
}
