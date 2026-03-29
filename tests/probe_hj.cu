// Probe: long long arithmetic, int64_t / uint64_t typedefs,
// mixing int and long long in expressions

#include <stdint.h>

__global__ void ll_arith(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] * b[tid] + a[tid] - b[tid];
    }
}

__global__ void u64_arith(uint64_t *out, uint64_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint64_t v = in[tid];
        // Hash-like scramble
        v ^= v >> 33;
        v *= 0xff51afd7ed558ccdULL;
        v ^= v >> 33;
        out[tid] = v;
    }
}

__global__ void mixed_int_ll(long long *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Widen 32-bit values for 64-bit multiply
        long long la = (long long)a[tid];
        long long lb = (long long)b[tid];
        out[tid] = la * lb;
    }
}
