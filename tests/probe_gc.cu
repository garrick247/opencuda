// Probe: volatile shared memory (common in reductions), memory fences,
// __threadfence, __threadfence_block, __threadfence_system

__global__ void threadfence_pattern(float *out, float *in, int *flag, int n) {
    int tid = threadIdx.x;

    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
    __threadfence();  // ensure stores visible to other blocks

    if (tid == 0) {
        atomicAdd(flag, 1);
        __threadfence_system();  // ensure visible to CPU
    }
}

__global__ void volatile_reduce(float *out, volatile float *sdata, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        sdata[tid] = in[tid];
    }
    __syncthreads();
    if (tid < n) {
        out[tid] = sdata[tid];
    }
}

// __threadfence_block
__global__ void block_fence(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    if (tid < n && tid < 256) {
        smem[tid] = in[tid];
    }
    __threadfence_block();
    __syncthreads();
    if (tid < n && tid < 256) {
        out[tid] = smem[(tid + 1) % 256];
    }
}
