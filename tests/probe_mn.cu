// Probe: Integer type edge cases
// - uint64_t / long long arithmetic
// - Mixed signed/unsigned 64-bit operations
// - Bitfield-like masking patterns
// - Pointer cast via (uintptr_t) style
// - Large literal constants (> 2^31)
// - 64-bit shift operations
// - Integer overflow wrap-around patterns

__global__ void u64_arithmetic(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long a = in[tid];
        unsigned long long b = in[n - 1 - tid];
        out[tid] = a * b + a - b;
    }
}

__global__ void i64_arithmetic(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        out[tid] = v * v - v + 1LL;
    }
}

// Mixed 32/64 bit
__global__ void mixed_width(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        long long wide = (long long)x * (long long)x;
        out[tid] = wide + (long long)x;
    }
}

// Large literal
__global__ void large_literal(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int mask = 0xFFFF0000u;
        unsigned int val = (unsigned int)tid;
        out[tid] = (val & mask) | ((val * 7u) & 0x0000FFFFu);
    }
}

// 64-bit shift
__global__ void u64_shift(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        out[tid] = (v << 3) | (v >> 61);  // rotate left 3
    }
}

// Pointer arithmetic using 64-bit index
__global__ void u64_index(float *out, float *in, long long stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long idx = (long long)tid * stride;
        out[tid] = in[idx % n];
    }
}

// Bit manipulation pattern
__global__ void bit_manipulation(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Count set bits (naive)
        int count = 0;
        for (int i = 0; i < 32; i++) {
            count += (v >> i) & 1u;
        }
        out[tid] = (unsigned int)count;
    }
}
