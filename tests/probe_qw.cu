// Probe: warp-level primitives with struct results, __ballot_sync
// patterns, complex mask operations, and shared memory bank patterns.

// ------------------------------------------------------------------
// __shfl_down_sync with float: warp reduction.

__global__ void warp_reduce_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        // Warp reduction using shfl_down
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        if (tid % 32 == 0) {
            out[tid / 32] = val;
        }
    }
}

// ------------------------------------------------------------------
// __ballot_sync: count set bits in a predicate across warp.

__global__ void ballot_count(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // Count threads in warp where v > 0
        unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
        if (tid % 32 == 0) {
            out[tid / 32] = __popc(mask);
        }
    }
}

// ------------------------------------------------------------------
// __shfl_xor_sync: butterfly reduction for sum.

__global__ void butterfly_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
        out[tid] = val;
    }
}

// ------------------------------------------------------------------
// __shfl_up_sync: prefix sum (scan) within a warp.

__global__ void warp_scan(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    if (tid < n) {
        float val = in[tid];
        float tmp;
        tmp = __shfl_up_sync(0xFFFFFFFF, val, 1);  if (lane >= 1)  val += tmp;
        tmp = __shfl_up_sync(0xFFFFFFFF, val, 2);  if (lane >= 2)  val += tmp;
        tmp = __shfl_up_sync(0xFFFFFFFF, val, 4);  if (lane >= 4)  val += tmp;
        tmp = __shfl_up_sync(0xFFFFFFFF, val, 8);  if (lane >= 8)  val += tmp;
        tmp = __shfl_up_sync(0xFFFFFFFF, val, 16); if (lane >= 16) val += tmp;
        out[tid] = val;
    }
}

// ------------------------------------------------------------------
// __shfl_sync on int: broadcast lane 0 to all.

__global__ void broadcast_lane0(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        int bcast = __shfl_sync(0xFFFFFFFF, val, 0);
        out[tid] = bcast;
    }
}

// ------------------------------------------------------------------
// Shared memory with stride-1 and stride-32 access (bank conflict test).

__global__ void smem_stride(float *out, float *in, int n) {
    __shared__ float smem[64];
    int tid = threadIdx.x;
    int lane = tid % 32;
    // Stride-1: conflict-free
    if (tid < 64) smem[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    // Access with offset by lane
    if (tid < n && tid < 32) {
        int idx = (lane + 1) % 32;
        out[tid] = smem[idx] + smem[idx + 32];
    }
}
