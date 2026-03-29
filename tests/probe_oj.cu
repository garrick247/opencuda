// Probe: struct with 64-bit fields, mixed-width struct offsets,
// unsigned overflow arithmetic, large constant expressions.

// ------------------------------------------------------------------
// Struct with mixed 32/64-bit fields: offset of 64-bit fields must be
// 8-byte aligned per C struct layout rules.
// Layout: { int a (4), <4 pad>, long long b (8), float c (4), <4 pad>, double d (8) }
// Actually C doesn't insert padding between fields unless needed for alignment.
// Simplified: { int a (4), int b (4), long long c (8), double d (8) } = 24 bytes

struct Mixed64 {
    int   a;
    int   b;
    long long c;
    double    d;
};

__global__ void mixed64_layout(double *out, int ia, int ib, long long ic, double id) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Mixed64 s;
        s.a = ia;
        s.b = ib;
        s.c = ic;
        s.d = id;
        // Verify field access round-trips correctly
        out[0] = (double)s.a;
        out[1] = (double)s.b;
        out[2] = (double)s.c;
        out[3] = s.d;
    }
}

// ------------------------------------------------------------------
// Pointer to struct with 64-bit fields: arrow access.

__global__ void ptr_mixed64(double *out, Mixed64 *p) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = (double)p->a;
        out[1] = (double)p->b;
        out[2] = (double)p->c;
        out[3] = p->d;
    }
}

// ------------------------------------------------------------------
// Unsigned arithmetic: wrapping on uint32 subtraction.
// (uint32)(0u - 1u) == 0xFFFFFFFF — no undefined behavior in C.

__global__ void uint_wrap(unsigned int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int x = 0u;
        unsigned int y = x - 1u;      // wraps to 0xFFFFFFFF
        unsigned int z = y + 1u;      // wraps back to 0
        out[0] = y;
        out[1] = z;
    }
}

// ------------------------------------------------------------------
// Integer literal suffixes: UL, LL, ULL.
// Tests that 64-bit literals are emitted correctly.

__global__ void literal_suffixes(long long *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        long long a = 0x7FFFFFFFLL;
        long long b = 0x80000000LL;
        unsigned long long c = 0xFFFFFFFFULL;
        unsigned long long d = 1ULL << 40;
        out[0] = a;
        out[1] = b;
        out[2] = (long long)c;
        out[3] = (long long)d;
    }
}
