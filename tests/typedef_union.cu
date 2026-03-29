// Regression: typedef union { ... } Alias; — typedef with inline union body
// Without fix: _parse_typedef only checked KW_STRUCT, not KW_UNION →
//   fell through to _parse_type_with_ptr → ParseError "expected IDENT, got LBRACE '{'".
// Fix: typedef check extended to KW_STRUCT || KW_UNION before falling through.

typedef union {
    float f;
    int i;
    unsigned int u;
} FloatBits;

typedef union {
    int lo;
    int hi;
} IntPair;

__global__ void float_bits_xor(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        FloatBits fb;
        fb.f = in[tid];
        // Flip sign bit via union — accesses scalar fields
        fb.u = fb.u ^ 0x80000000u;
        out[tid] = fb.f;
    }
}

__global__ void union_int(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        IntPair p;
        p.lo = in[tid];
        p.hi = in[tid] + 1;
        out[tid] = p.lo + p.hi;
    }
}
