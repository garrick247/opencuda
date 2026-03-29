// Probe: C operator precedence gotchas,
// signed vs unsigned right shift (arithmetic vs logical),
// unary minus on expressions,
// bitwise precedence (^ tighter than |, & tighter than ^)

// Precedence: << lower than +
// a << (2 + 1) is correct C, NOT (a << 2) + 1
__global__ void shift_plus_prec(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        // In C: a << 2 + 1  means  a << (2+1) = a << 3  (shift takes lower prec)
        int v1 = a << 2 + 1;          // = a * 8
        int v2 = (a << 2) + 1;        // = a * 4 + 1
        out[tid * 2]     = v1;
        out[tid * 2 + 1] = v2;
    }
}

// Precedence: & lower than + (common gotcha)
// a & 0xF + 1  means  a & (0xF + 1)  = a & 16  NOT  (a & 0xF) + 1
__global__ void and_plus_prec(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int v1 = a & 0xF + 1;         // = a & 16 (0x10)  — & has lower prec than +
        int v2 = (a & 0xF) + 1;       // = lower nibble + 1
        out[tid * 2]     = v1;
        out[tid * 2 + 1] = v2;
    }
}

// Precedence: ^ tighter than |
// a | b ^ c  means  a | (b ^ c)
__global__ void or_xor_prec(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid * 3];
        int b = in[tid * 3 + 1];
        int c = in[tid * 3 + 2];
        int v1 = a | b ^ c;            // = a | (b ^ c)
        int v2 = (a | b) ^ c;
        out[tid * 2]     = v1;
        out[tid * 2 + 1] = v2;
    }
}

// Signed vs unsigned right shift
__global__ void signed_unsigned_shr(int *out, int *in_s, unsigned int *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = in_s[tid];
        unsigned int u = in_u[tid];
        // signed right shift: arithmetic (sign-extending) → shr.s32
        // unsigned right shift: logical (zero-fill)       → shr.u32
        out[tid * 2]     = s >> 4;     // sign-extending if negative
        out[tid * 2 + 1] = (int)(u >> 4);  // zero-filling
    }
}

// Unary minus on complex expressions
__global__ void unary_minus_expr(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid];
        int vb = b[tid];
        int v1 = -(va + vb);           // negate sum
        int v2 = -va + vb;             // negate first then add  (= -(va) + vb)
        int v3 = -(va * vb);           // negate product
        out[tid * 3]     = v1;
        out[tid * 3 + 1] = v2;
        out[tid * 3 + 2] = v3;
    }
}
