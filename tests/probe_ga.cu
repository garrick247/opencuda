// Probe: 64-bit shift with 32-bit shift amount (common pattern)
// PTX shl.b64 / shr.b64 requires the shift amount to be 32-bit, but
// the value being shifted to be 64-bit — check that codegen emits correctly

__global__ void shl64_with_s32_amt(long long *out, long long *in, int *amounts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        int amt = amounts[tid] & 63;
        // 64-bit value shifted by 32-bit amount
        out[tid] = v << amt;
    }
}

__global__ void shr64_arithmetic(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        // Arithmetic right shift — should preserve sign bit
        out[tid] = v >> 1;
    }
}

__global__ void shr64_logical(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        // Logical right shift
        out[tid] = v >> 1;
    }
}

// Mixed 64/32-bit in complex expression
__global__ void mixed64(long long *out, int *in32, long long *in64, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long big = in64[tid];
        int small = in32[tid];
        // Widening arithmetic
        long long result = (big >> small) + ((long long)small << 32);
        out[tid] = result;
    }
}
