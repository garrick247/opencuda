// Regression: full set of compound assignment operators /=, %=, &=, |=, ^=, <<=, >>=
// Without fix: ParseError "expected SEMI, got SLASH_EQ '/='" (and similar for others)
// Fix: _parse_stmt compound-assign list extended to include all C compound operators

__global__ void compound_div_mod(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        v /= 3;
        v %= 7;
        out[tid] = v;
    }
}

__global__ void compound_bitwise(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        v &= 0x0000FFFFu;
        v |= 0x80000000u;
        v ^= 0x5A5A5A5Au;
        v <<= 2;
        v >>= 1;
        out[tid] = v;
    }
}

__global__ void compound_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        v /= 4.0f;
        v *= 1.5f;
        out[tid] = v;
    }
}

__global__ void compound_array_elem(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] /= 2;
        out[tid] += in[tid];
        out[tid] &= 0xFF;
    }
}
