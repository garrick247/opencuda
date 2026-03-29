// Probe: unusual but valid C patterns that might trip up parsing
// - Unary minus applied to function result
// - Complex array init with expressions (not constants)
// - Negation of unsigned
// - Bitfield-style masking patterns
// - Multiple assignment operators in one expression (a = b = expr)

__global__ void unary_patterns(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int neg = -v;
        int bit7 = (v >> 7) & 1;
        int lo8  = v & 0xFF;
        int hi8  = (v >> 8) & 0xFF;
        int sign = (v < 0) ? -1 : 1;
        out[tid] = neg + bit7 + lo8 + hi8 + sign;
    }
}

__global__ void assign_chain(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = b = c = tid * 3 + 1;
        out[tid] = a + b + c;
    }
}

// Logical NOT in expression
__global__ void logical_not(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int is_zero  = !v;
        int not_sign = !(v < 0);
        out[tid] = is_zero + not_sign * 2;
    }
}
