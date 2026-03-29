// Probe: warp-level primitives — __match_any_sync, __match_all_sync,
// __reduce_add_sync, __reduce_min_sync, __reduce_max_sync (Ampere+)
// Also: lane masking patterns with FULL_MASK

#define FULL_MASK 0xffffffff

__global__ void warp_reduce_add(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        // Manual warp reduction using shfl_down
        val += __shfl_down_sync(FULL_MASK, val, 16);
        val += __shfl_down_sync(FULL_MASK, val, 8);
        val += __shfl_down_sync(FULL_MASK, val, 4);
        val += __shfl_down_sync(FULL_MASK, val, 2);
        val += __shfl_down_sync(FULL_MASK, val, 1);
        if ((tid & 31) == 0) {
            atomicAdd(&out[0], val);
        }
    }
}

// __any_sync / __all_sync
__global__ void warp_vote_sync(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int pred = in[tid] > 0;
        int any = __any_sync(FULL_MASK, pred);
        int all = __all_sync(FULL_MASK, pred);
        out[tid] = any + all;
    }
}

// __shfl_xor_sync pattern (butterfly reduction)
__global__ void butterfly_xor(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        val ^= __shfl_xor_sync(FULL_MASK, val, 1);
        val ^= __shfl_xor_sync(FULL_MASK, val, 2);
        val ^= __shfl_xor_sync(FULL_MASK, val, 4);
        out[tid] = val;
    }
}
