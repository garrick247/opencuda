// Probe: multiple __shared__ arrays in same kernel,
// mixed shared scalar and array,
// __shared__ struct, shared memory pointer aliasing

struct SharedBuf {
    float sum;
    int count;
};

__global__ void multi_shared(float *out, float *in, int n) {
    __shared__ float smem_a[32];
    __shared__ float smem_b[32];
    __shared__ int   smem_cnt[32];

    int tid = threadIdx.x;
    if (tid < 32) {
        smem_a[tid] = (tid < n) ? in[tid] : 0.0f;
        smem_b[tid] = (tid < n) ? in[tid] * 2.0f : 0.0f;
        smem_cnt[tid] = (tid < n) ? 1 : 0;
        __syncthreads();
        out[tid] = smem_a[tid] + smem_b[tid] + (float)smem_cnt[tid];
    }
}

// __shared__ scalar (not array)
__global__ void shared_scalar(float *out, float *in, int n) {
    __shared__ float total;
    int tid = threadIdx.x;
    if (tid == 0) {
        total = 0.0f;
        for (int i = 0; i < n; i++) total += in[i];
    }
    __syncthreads();
    if (tid < n) out[tid] = total;
}

// Warp-level reduction using shared + shuffle
__global__ void warp_shared_reduce(float *out, float *in, int n) {
    __shared__ float warp_results[32];
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    // Warp reduce
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    int lane = tid & 31;
    if (lane == 0) {
        warp_results[tid >> 5] = val;
    }
    __syncthreads();
    if (tid == 0) out[0] = warp_results[0];
}
