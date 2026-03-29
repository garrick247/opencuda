// Probe: 64-bit shift, 64-bit comparison, mixed s64/u64 arithmetic
// Also: negative 64-bit constants, 64-bit modulo

__global__ void u64_shr(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        // Logical right shift on 64-bit
        out[tid] = (v >> 33) ^ (v << 31);
    }
}

__global__ void s64_arith(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        // Arithmetic ops on signed 64-bit
        long long a = v * 1000000007LL;
        long long b = a - v * 998244353LL;
        long long c = b % 1000000007LL;
        out[tid] = c;
    }
}

// 64-bit loop counter
__global__ void large_loop(float *out, float *in, long long n) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        long long end = tid + 4LL;
        for (long long i = tid; i < end && i < n; i++) {
            sum += in[i];
        }
        out[tid] = sum;
    }
}
