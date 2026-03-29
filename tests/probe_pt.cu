// Probe: type coercion, implicit conversions, signed/unsigned mixing,
// narrowing casts, and selp/predicate interactions.

// ------------------------------------------------------------------
// Signed/unsigned comparison: int vs unsigned int.
// Mixing signed and unsigned in the same comparison.

__global__ void signed_unsigned_cmp(int *out, int *sa, unsigned int *ua, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sv = sa[tid];
        unsigned int uv = ua[tid];
        // Implicit: compare signed to unsigned — C promotes to unsigned
        out[tid] = (sv < (int)uv) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Unsigned arithmetic: prevent sign extension on add/shift.

__global__ void unsigned_arith(unsigned int *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = data[tid];
        // Shift right must be logical (zero-fill), not arithmetic
        unsigned int shifted = v >> 1;
        unsigned int masked  = v & 0x7FFFFFFF;
        out[tid] = shifted + masked;
    }
}

// ------------------------------------------------------------------
// Narrowing cast: int → short → char → int round-trip.
// Tests that truncation is emitted and sign-extends on widening.

__global__ void narrow_cast(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        short s = (short)v;       // truncate to 16-bit
        char  c = (char)s;        // truncate to 8-bit
        int   r = (int)c;         // sign-extend back to 32-bit
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Unsigned narrowing: uint → uchar, no sign extension.

__global__ void unsigned_narrow(unsigned int *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = data[tid];
        unsigned char uc = (unsigned char)v;
        out[tid] = (unsigned int)uc;  // zero-extend, not sign-extend
    }
}

// ------------------------------------------------------------------
// Float-to-int with explicit cast: truncation toward zero.
// (int)3.7f → 3, (int)(-3.7f) → -3

__global__ void float_to_int_trunc(int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        int r = (int)v;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Int-to-float with round-to-nearest.
// Large int (2^24+1) may lose precision when cast to float.

__global__ void int_to_float_rn(float *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = (float)data[tid];
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Unsigned long long → int narrowing.
// Tests that u64 → u32 truncation is explicit.

__global__ void ull_to_int(int *out, unsigned long long *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = data[tid];
        int r = (int)(v & 0xFFFFFFFF);  // low 32 bits
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Ternary with mixed int/unsigned result.
// The result type should follow C's usual arithmetic conversions.

__global__ void ternary_mixed_type(int *out, int *cond_data,
                                   int *iv, unsigned int *uv, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // C promotes to unsigned when mixing signed/unsigned in ternary
        int c = cond_data[tid];
        int r = (int)((c > 0) ? (unsigned int)iv[tid] : uv[tid]);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Predicate (bool-like) used in arithmetic.
// `int flag = (x > 0); out = flag * x;` — flag is 0 or 1.

__global__ void pred_in_arith(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int flag = (v > 0) ? 1 : 0;
        out[tid] = flag * v;
    }
}

// ------------------------------------------------------------------
// Chained narrowing and widening with arithmetic.
// Tests the full CVT chain: s32→u8→s32 within arithmetic.

__global__ void chained_cvt(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // Saturate to byte range via unsigned char cast, then scale
        unsigned char lo = (unsigned char)(v & 0xFF);
        unsigned char hi = (unsigned char)((v >> 8) & 0xFF);
        out[tid] = (int)lo + (int)hi * 256;
    }
}
