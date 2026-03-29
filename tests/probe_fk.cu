// Probe: shr.b64, bitwise on 64-bit, 64-bit AND/OR/XOR patterns
// Also: popcount on 64-bit, bit reversal, gray code

__global__ void bit64_ops(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        // Various 64-bit bitwise ops
        unsigned long long a = v & 0xAAAAAAAAAAAAAAAAULL;
        unsigned long long b = v | 0x5555555555555555ULL;
        unsigned long long c = v ^ 0xFFFFFFFFFFFFFFFFULL;
        unsigned long long d = ~v;
        unsigned long long e = a & ~b;
        out[tid] = (a ^ b ^ c ^ d ^ e);
    }
}

__global__ void gray_code(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        // Gray code encode
        out[tid] = v ^ (v >> 1);
    }
}

// Mix of 32-bit and 64-bit in same kernel
__global__ void mixed_width(long long *out, int *in32, long long *in64, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in32[tid];
        long long b = in64[tid];
        // Widen and combine
        long long c = (long long)a * b;
        long long d = c + (long long)a;
        out[tid] = d >> 1;
    }
}
