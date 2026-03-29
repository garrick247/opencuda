// Probe: Special CUDA patterns - warp-level, block-level cooperation
// - __activemask()
// - __popc() (population count intrinsic)
// - __clz() (count leading zeros)
// - __ffs() (find first set)
// - lane_id via threadIdx.x & 31

__global__ void warp_primitives(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int lane = tid & 31;
        // Warp vote
        unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
        int count = 0;
        unsigned int m = mask;
        while (m) { count += (int)(m & 1u); m >>= 1; }
        // Leader lane broadcasts
        int leader_val = __shfl_sync(0xFFFFFFFF, v, 0);
        out[tid] = count + leader_val + lane;
    }
}

__global__ void bit_intrinsics(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Simulate __popc manually (real CUDA uses intrinsic)
        int popcount = 0;
        unsigned int tmp = v;
        while (tmp) { popcount++; tmp &= (tmp - 1u); }
        // Simulate __clz: find highest set bit
        int clz = 0;
        if (v != 0) {
            unsigned int bit = 0x80000000u;
            while (!(v & bit)) { clz++; bit >>= 1; }
        } else {
            clz = 32;
        }
        out[tid] = popcount + clz;
    }
}
