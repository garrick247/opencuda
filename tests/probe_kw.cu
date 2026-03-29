// Probe: warp shuffle patterns, __ballot_sync, __activemask,
// reduction via shuffle-down, predicated warp ops

// Warp-level sum reduction via shuffle-down
__global__ void warp_reduce_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;

    // Reduce within warp (assumes warp size = 32)
    unsigned mask = 0xFFFFFFFFu;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    if (tid == 0) {
        out[0] = val;
    }
}

// __shfl_xor_sync for butterfly reduction
__global__ void warp_xor_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    unsigned mask = __activemask();

    val ^= __shfl_xor_sync(mask, val, 16);
    val ^= __shfl_xor_sync(mask, val, 8);
    val ^= __shfl_xor_sync(mask, val, 4);
    val ^= __shfl_xor_sync(mask, val, 2);
    val ^= __shfl_xor_sync(mask, val, 1);

    if (tid == 0) {
        out[0] = val;
    }
}

// __ballot_sync: count threads with positive value
__global__ void ballot_count(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    unsigned mask = 0xFFFFFFFFu;

    unsigned ballot = __ballot_sync(mask, val > 0);
    int count = __popc(ballot);

    if (tid == 0) {
        out[0] = count;
    }
}

// __shfl_up_sync: prefix sum within warp (simple version)
__global__ void warp_prefix(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        unsigned mask = 0xFFFFFFFFu;

        int t1 = __shfl_up_sync(mask, val, 1);
        if (tid >= 1) val += t1;
        int t2 = __shfl_up_sync(mask, val, 2);
        if (tid >= 2) val += t2;
        int t4 = __shfl_up_sync(mask, val, 4);
        if (tid >= 4) val += t4;

        out[tid] = val;
    }
}

// __syncthreads in shared-memory pattern
__global__ void syncthreads_reduce(int *out, int *in, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;

    sdata[tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();

    // Two-step binary tree reduce
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)  sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32)  sdata[tid] += sdata[tid + 32];
    __syncthreads();

    if (tid == 0) {
        out[0] = sdata[0];
    }
}
