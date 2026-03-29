// Probe: warp-level primitives — __ballot_sync, __activemask,
// __all_sync, __any_sync, warp reduction with __shfl_down_sync

#define FULL_MASK 0xFFFFFFFF

__global__ void warp_ballot(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        unsigned mask = __ballot_sync(FULL_MASK, val > 0);
        out[tid] = (int)__popc(mask);
    }
}

__global__ void warp_any_all(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        int any_pos = __any_sync(FULL_MASK, val > 0);
        int all_pos = __all_sync(FULL_MASK, val > 0);
        out[tid] = any_pos * 2 + all_pos;
    }
}

__global__ void warp_reduce_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if (tid % 32 == 0) {
            out[tid / 32] = val;
        }
    }
}
