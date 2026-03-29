// Probe: long long arithmetic, 64-bit indices, large array offsets
// Also: unsigned long long literals and arithmetic

__global__ void large_index(float *out, float *in, long long n, long long stride) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        long long idx = tid * stride;
        out[tid] = in[idx];
    }
}

__global__ void u64_arith(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        v = v * 6364136223846793005ULL + 1442695040888963407ULL;
        out[tid] = v;
    }
}

// 64-bit bit operations
__global__ void u64_bits(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        v = (v >> 32) ^ (v & 0xFFFFFFFFULL);
        v = v * 0x9e3779b97f4a7c15ULL;
        out[tid] = v;
    }
}
