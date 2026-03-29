// Probe: signed vs unsigned comparison semantics,
// unsigned overflow wrapping, signed-to-unsigned and back,
// unsigned right shift vs signed right shift,
// unsigned division and modulo

// Unsigned comparison: values > INT_MAX should be compared as unsigned
__global__ void unsigned_cmp(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // These branch on the MSB being set
        unsigned int a = (v > 2147483648u) ? 1u : 0u;    // > 2^31
        unsigned int b = (v >= 0xFFFFFFFFu) ? 1u : 0u;   // >= UINT_MAX
        out[tid * 2]     = a;
        out[tid * 2 + 1] = b;
    }
}

// Unsigned arithmetic: wrapping at 2^32
__global__ void unsigned_wrap(unsigned int *out, unsigned int a, unsigned int b) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int sum  = a + b;          // wraps
        unsigned int diff = a - b;          // wraps if b > a
        unsigned int prod = a * b;          // wraps
        out[0] = sum;
        out[1] = diff;
        out[2] = prod;
    }
}

// Logical right shift on unsigned vs arithmetic on signed
__global__ void shift_sign(int *out, unsigned int *uout, int x, unsigned int u) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Signed: arithmetic shift right (fills with sign bit)
        int sx = x >> 4;         // shr.s32 — sign-extending
        // Unsigned: logical shift right (fills with zero)
        unsigned int ux = u >> 4;  // shr.b32 — zero-filling
        out[0]  = sx;
        uout[0] = ux;
    }
}

// Unsigned division and modulo
__global__ void unsigned_divmod(unsigned int *out, unsigned int a, unsigned int b) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int q = a / b;
        unsigned int r = a % b;
        out[0] = q;
        out[1] = r;
    }
}

// Mixed signed/unsigned: signed < unsigned comparison requires care
__global__ void mixed_sign_arith(int *out, int x, unsigned int u) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Casting between signed and unsigned
        unsigned int xu = (unsigned int)x;
        int ui = (int)u;
        out[0] = (int)xu;   // round-trip
        out[1] = ui;
        out[2] = x + (int)u;  // explicit cast before add
    }
}
