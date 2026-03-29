// Probe: 64-bit AND/OR/XOR with 32-bit operands — verify codegen widens correctly
// Also: 64-bit comparison with mixed-width operands

__global__ void bit64_mixed(unsigned long long *out, unsigned int *in32,
                              unsigned long long *in64, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long big = in64[tid];
        unsigned int small = in32[tid];
        // AND: 64-bit value with 32-bit value (zero-extended)
        unsigned long long masked = big & (unsigned long long)small;
        unsigned long long ored = big | (unsigned long long)small;
        unsigned long long xored = big ^ (unsigned long long)small;
        out[tid] = masked + ored + xored;
    }
}

// 64-bit comparison — mixed s64 and u32
__global__ void cmp64_mixed(int *out, long long *in64, int *in32, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long big = in64[tid];
        int small = in32[tid];
        // Compare 64-bit with widened 32-bit
        if (big > (long long)small) {
            out[tid] = 1;
        } else if (big < (long long)small) {
            out[tid] = -1;
        } else {
            out[tid] = 0;
        }
    }
}

// 64-bit loop with 32-bit loop variable
__global__ void loop64_32_mix(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long sum = 0LL;
        for (int i = 0; i < n; i++) {
            sum += in[(long long)tid * n + i];
        }
        out[tid] = sum;
    }
}
