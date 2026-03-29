// Probe: integer overflow/wrap, signed vs unsigned comparison,
// narrow type arithmetic edge cases, and mixed-sign operations.

// ------------------------------------------------------------------
// Signed integer overflow (wraps in C, PTX uses same bits).

__global__ void int_overflow_wrap(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = 2147483647;  // INT_MAX
        int y = x + 1;       // wraps to INT_MIN in two's complement
        out[tid] = y;        // expected: -2147483648
    }
}

// ------------------------------------------------------------------
// Unsigned arithmetic: no overflow surprise.

__global__ void uint_no_overflow(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int x = 0xFFFFFFFFu;
        unsigned int y = x + 1u;    // wraps to 0
        unsigned int z = x + 2u;    // wraps to 1
        out[tid] = y + z;           // 0 + 1 = 1
    }
}

// ------------------------------------------------------------------
// Signed to unsigned promotion in comparison.

__global__ void signed_unsigned_compare(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Cast to unsigned for bitwise operations
        unsigned int uv = (unsigned int)v;
        // High bit test
        int is_negative = (int)(uv >> 31);
        out[tid] = is_negative;
    }
}

// ------------------------------------------------------------------
// Byte extraction and packing.

__global__ void byte_pack(unsigned int *out, unsigned char *bytes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Pack 4 bytes into a 32-bit word (little-endian)
        unsigned int b0 = (unsigned int)bytes[tid * 4 + 0];
        unsigned int b1 = (unsigned int)bytes[tid * 4 + 1];
        unsigned int b2 = (unsigned int)bytes[tid * 4 + 2];
        unsigned int b3 = (unsigned int)bytes[tid * 4 + 3];
        out[tid] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

// ------------------------------------------------------------------
// Float to int conversion with clamping.

__global__ void float_to_int_clamp(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Clamp to valid int range before conversion
        float clamped = v;
        if (clamped > 2147483520.0f)  clamped = 2147483520.0f;
        if (clamped < -2147483648.0f) clamped = -2147483648.0f;
        out[tid] = (int)clamped;
    }
}

// ------------------------------------------------------------------
// Double precision comparison (equality is risky, but < is fine).

__global__ void double_compare(int *out, double *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double va = a[tid];
        double vb = b[tid];
        int r = 0;
        if (va < vb)  r = -1;
        else if (va > vb) r = 1;
        // else: r = 0 (equal or NaN)
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float accumulation order sensitivity test.

__global__ void float_accum_order(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Forward accumulation
        float fwd = 0.0f;
        for (int i = 0; i < 8; i++) {
            fwd += in[tid * 8 + i];
        }
        // Backward accumulation (different order → potentially different result due to FP)
        float bwd = 0.0f;
        for (int i = 7; i >= 0; i--) {
            bwd += in[tid * 8 + i];
        }
        out[tid * 2]     = fwd;
        out[tid * 2 + 1] = bwd;
    }
}
