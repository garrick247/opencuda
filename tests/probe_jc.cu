// Probe: type promotion edge cases that produce silent wrong results,
// unsigned arithmetic wrapping,
// signed/unsigned comparison traps,
// large constants and bit pattern correctness

// unsigned subtraction wrapping: (uint)(0 - 1) = 0xFFFFFFFF
__global__ void uint_wrap(unsigned int *out, unsigned int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int x = 0u;
        unsigned int y = x - 1u;   // should wrap to 0xFFFFFFFF
        out[0] = y;
        out[1] = y >> 28;          // should be 15 (0xF)
        out[2] = (y & 0xF);        // should be 15
    }
}

// INT32_MIN negation trap: -INT32_MIN == INT32_MIN (overflow)
__global__ void int_min_trap(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = -2147483648;       // INT32_MIN
        out[0] = x;
        out[1] = -x;               // still INT32_MIN due to overflow
        out[2] = x >> 1;           // arithmetic right shift: -1073741824
        out[3] = x < 0 ? 1 : 0;   // should be 1
    }
}

// Mixed signed/unsigned comparison: -1 as signed is < 0, as unsigned is > UINT32_MAX
__global__ void sign_cmp(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int   s = -1;
        unsigned int u = 0xFFFFFFFFu;
        out[0] = (s < 0) ? 1 : 0;         // 1: signed comparison
        out[1] = (u > 0u) ? 1 : 0;        // 1: unsigned comparison
        out[2] = (s == (int)u) ? 1 : 0;   // 1: both 0xFFFFFFFF bit pattern
    }
}

// Bit pattern manipulation: float bits as int
__global__ void float_bits(int *out, float *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // 1.0f is 0x3F800000
        float f = in[0];
        // Extract exponent: bits 30-23
        int bits = *((int*)&f);  // reinterpret cast — may not be supported
        out[0] = bits;
    }
}
