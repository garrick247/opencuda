// Regression: volatile type qualifier must be silently accepted.
// PTX has no volatile qualifier — all ld/st are emitted as-is anyway.
// Without fix: ParseError "expected type, got 'volatile'"
__global__ void volatile_param_test(volatile float *out, volatile float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        volatile float v = in[tid];
        out[tid] = v * 2.0f;
    }
}

__global__ void volatile_shared_test(float *out, float *in, int n) {
    volatile __shared__ float smem[64];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = in[tid];
        __syncthreads();
        out[tid] = smem[63 - tid];
    }
}
