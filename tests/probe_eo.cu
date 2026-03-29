// Probe: volatile loads/stores, restrict pointers, const parameters
// __restrict__ is a CUDA performance hint that should parse without error

__global__ void restrict_add(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b,
                              int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// volatile pointer parameter
__global__ void volatile_store(volatile int *flag, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        flag[0] = n;
    }
}

// const array parameter (read-only)
__global__ void const_param(float *out, const float *lut, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = lut[idx[tid]];
    }
}
