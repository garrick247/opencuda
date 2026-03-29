// Probe: CUDA cooperative groups / warp-level APIs beyond basic shuffles,
// also: __activemask(), lane/warp utility patterns

#define FULL_MASK 0xFFFFFFFFU

__device__ int warp_prefix_sum(int val) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31;
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(mask, val, d);
        if (lane >= d) val += t;
    }
    return val;
}

__global__ void prefix_sum_warp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = warp_prefix_sum(in[tid]);
    }
}

// Warp vote + conditional broadcast
__global__ void warp_vote_bcast(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        int any = __any_sync(FULL_MASK, val > 0);
        int all = __all_sync(FULL_MASK, val > 0);
        // Broadcast lane 0's value to all lanes
        int lane0_val = __shfl_sync(FULL_MASK, val, 0);
        out[tid] = any * 10 + all * 100 + lane0_val;
    }
}

// Lane index utilities
__global__ void lane_utils(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lane = tid & 31;
        int warp = tid >> 5;
        // XOR shuffle (butterfly)
        int val = tid;
        for (int i = 1; i < 32; i <<= 1) {
            val += __shfl_xor_sync(FULL_MASK, val, i);
        }
        out[tid] = val + lane + warp;
    }
}
