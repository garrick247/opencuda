// Probe: type coercion edge cases, conditional stores, non-obvious
// operand widths, and expression patterns that might confuse type inference.

// ------------------------------------------------------------------
// Implicit int-to-float and float-to-int conversions in assignments.

__global__ void coerce_assign(float *fout, int *iout, float *fin, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // int = float (truncation)
        iout[tid] = fin[tid];
        // float = int (promotion)
        fout[tid] = iin[tid];
    }
}

// ------------------------------------------------------------------
// Unsigned vs signed comparison: u32 vs s32 generates different setp.

__global__ void unsigned_cmp(int *out, unsigned int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // unsigned vs signed comparison
        unsigned int ua = a[tid];
        int sb = b[tid];
        // ua < sb: ua is unsigned, sb is signed — signed wins in C
        out[tid] = (ua < (unsigned int)sb) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Conditional assignment: x = cond ? a : b where a and b have different types.

__global__ void cond_type_mix(float *out, int *cond, float *af, int *ai, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // ternary result: float vs int — result should be float
        float r = cond[tid] ? af[tid] : (float)ai[tid];
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Bit manipulation: extract, insert, rotate.

__global__ void bit_ops(unsigned int *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = data[tid];
        // Extract bits 4..11 (8-bit field)
        unsigned int field = (v >> 4) & 0xFF;
        // Insert bits 16..23
        unsigned int inserted = (v & ~(0xFFu << 16)) | (field << 16);
        // Rotate left by 8
        unsigned int rotated = (inserted << 8) | (inserted >> 24);
        out[tid] = rotated;
    }
}

// ------------------------------------------------------------------
// Float comparison and NaN behavior: fmin/fmax via ternary.

__global__ void float_minmax(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = a[tid], y = b[tid];
        float mn = x < y ? x : y;
        float mx = x > y ? x : y;
        out[tid * 2 + 0] = mn;
        out[tid * 2 + 1] = mx;
    }
}

// ------------------------------------------------------------------
// Pointer difference and byte-level pointer arithmetic.

__global__ void ptr_diff(int *out, char *base, char *ptr, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Pointer difference in bytes (ptrdiff)
        long long diff = (long long)(ptr - base);
        out[tid] = (int)(diff / 4);  // convert bytes to int index
    }
}

// ------------------------------------------------------------------
// Signed shift vs unsigned shift on negative values.

__global__ void shift_signedness(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int  sv = data[tid];
        unsigned int uv = (unsigned int)sv;
        // Signed right shift (arithmetic, sign-extends)
        int  sar = sv >> 3;
        // Unsigned right shift (logical, zero-extends)
        unsigned int lsr = uv >> 3;
        // Combine to produce deterministic output
        out[tid] = sar + (int)lsr;
    }
}

// ------------------------------------------------------------------
// Cast chain: double -> float -> int -> long long.

__global__ void cast_chain(long long *out, double *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double d = data[tid];
        float  f = (float)d;
        int    i = (int)f;
        long long ll = (long long)i;
        out[tid] = ll * ll;
    }
}
