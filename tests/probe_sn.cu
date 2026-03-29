// Probe: edge cases in type promotion — comparing pointers,
// function returning void*, NULL handling, and size_t patterns.

// ------------------------------------------------------------------
// Integer division with rounding patterns.

__global__ void div_round(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Ceiling division: (v + d - 1) / d
        int ceil2 = (v + 1) / 2;
        int ceil4 = (v + 3) / 4;
        int ceil8 = (v + 7) / 8;
        // Round to nearest: (v + d/2) / d
        int rnd2 = (v + 1) / 2;
        out[tid] = ceil2 + ceil4 + ceil8 + rnd2;
    }
}

// ------------------------------------------------------------------
// Modulo with negative numbers.

__global__ void signed_mod(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // C % is truncated toward zero — negative results possible
        int m3 = v % 3;
        int m7 = v % 7;
        int m256 = v % 256;
        // Wrap to positive: ((v % m) + m) % m
        int pos_m3 = ((v % 3) + 3) % 3;
        out[tid] = m3 + m7 + m256 + pos_m3;
    }
}

// ------------------------------------------------------------------
// Power of 2 check and manipulation.

__global__ void pow2_ops(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Is power of 2?
        int is_pow2 = (v != 0) && ((v & (v - 1)) == 0);
        // Round up to next power of 2
        unsigned int p = v - 1;
        p |= p >> 1; p |= p >> 2; p |= p >> 4;
        p |= p >> 8; p |= p >> 16;
        unsigned int next_pow2 = p + 1;
        out[tid] = is_pow2 + (int)next_pow2;
    }
}

// ------------------------------------------------------------------
// Leading zeros / trailing zeros / population count of complex exprs.

__global__ void bit_ops(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        int lz = __clz(v);
        int pc = __popc(v);
        int lz_mask = __clz(v | 1u);  // always defined (v|1 != 0)
        // Bit reversal
        unsigned int rev = __brev(v);
        out[tid] = (unsigned int)(lz + pc + lz_mask) + rev;
    }
}

// ------------------------------------------------------------------
// Fixed-point arithmetic: scale factor as integer fraction.

__global__ void fixed_point(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Multiply by 0.75 using integers: (v * 3) >> 2
        int x = (v * 3) >> 2;
        // Multiply by 0.5: v >> 1
        int y = v >> 1;
        // Multiply by 1.5: v + (v >> 1)
        int z = v + (v >> 1);
        out[tid] = x + y + z;
    }
}

// ------------------------------------------------------------------
// Saturation arithmetic (manual clamp).

__global__ void sat_arith(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        // Saturating add: clamp to [INT_MIN, INT_MAX]
        long long sum = (long long)av + (long long)bv;
        int sat = (sum > 2147483647LL) ? 2147483647 :
                  (sum < -2147483648LL) ? -2147483648 :
                  (int)sum;
        // Saturating multiply (just check sign)
        long long prod = (long long)av * (long long)bv;
        int sat_prod = (prod > 2147483647LL) ? 2147483647 :
                       (prod < -2147483648LL) ? -2147483648 :
                       (int)prod;
        out[tid] = sat + sat_prod;
    }
}

// ------------------------------------------------------------------
// Counting trailing zeros manually.

__global__ void ctz_manual(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // ctz via: __popc((v & (-v)) - 1)
        unsigned int lowest = v & (unsigned int)(-(int)v);
        int ctz = __popc(lowest - 1u);
        // Also: ~v & (v-1) gives ones below lowest set bit
        int alt = __popc(~v & (v - 1u));
        out[tid] = ctz + alt;
    }
}
