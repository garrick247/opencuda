// Probe: __launch_bounds__, template-like macros for block sizes,
// extern shared memory declaration, dynamic shared memory pattern

__global__ void __launch_bounds__(256, 2) bounded_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid];
    }
}

// Dynamic shared memory via extern __shared__
__global__ void dynamic_shmem(float *out, float *in, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    if (tid < n) {
        smem[tid] = in[tid];
    }
    __syncthreads();
    if (tid < n) {
        float sum = 0.0f;
        int lo = tid > 0 ? tid - 1 : 0;
        int hi = tid < bdim - 1 ? tid + 1 : bdim - 1;
        for (int i = lo; i <= hi; i++) {
            sum += smem[i];
        }
        out[tid] = sum;
    }
}

// Shared memory with explicit size
__global__ void explicit_shmem(float *out, float *in, int n) {
    __shared__ float buf[512];
    int tid = threadIdx.x;
    if (tid < n && tid < 512) {
        buf[tid] = in[tid] * in[tid];
        __syncthreads();
        out[tid] = buf[tid];
    }
}
