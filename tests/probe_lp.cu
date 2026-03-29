// Probe: 64-bit integer (long long) arithmetic,
// unsigned long long, mixed 32/64 bit ops,
// integer division by constant (strength reduction opportunity),
// size_t type usage

// 64-bit integer arithmetic
__global__ void int64_ops(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long a = in[tid * 2];
        long long b = in[tid * 2 + 1];
        out[tid * 3]     = a + b;
        out[tid * 3 + 1] = a * b;
        out[tid * 3 + 2] = a - b;
    }
}

// Unsigned 64-bit
__global__ void uint64_ops(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long a = in[tid];
        out[tid] = a * 2ULL + 1ULL;
    }
}

// Mixed 32/64-bit: 32-bit index into 64-bit array, address computation
__global__ void mixed_width(long long *out, int *indices, long long *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid];
        long long v = vals[idx];   // 64-bit load at 32-bit index
        out[tid] = v + (long long)tid;
    }
}

// Division by power of 2 — should strength-reduce to shift
__global__ void div_by_pow2(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid * 3]     = v / 2;    // → asr 1 (signed arithmetic right shift)
        out[tid * 3 + 1] = v / 4;    // → asr 2
        out[tid * 3 + 2] = v / 8;    // → asr 3
    }
}

// size_t (typically unsigned long long on 64-bit)
__global__ void size_t_ops(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        size_t total = (size_t)n * n;   // n^2
        out[0] = (int)total;
    }
}
