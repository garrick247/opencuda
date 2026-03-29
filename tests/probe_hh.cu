// Probe: __builtin_popcount, __builtin_clz, __builtin_ctz,
// __popc, __clz, __ffs (CUDA intrinsic integer bit ops)

__global__ void popcount_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __popc(in[tid]);
    }
}

__global__ void clz_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __clz(in[tid]);
    }
}

__global__ void ffs_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ffs(in[tid]);
    }
}

// brev (bit reverse)
__global__ void brev_kernel(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __brev(in[tid]);
    }
}
