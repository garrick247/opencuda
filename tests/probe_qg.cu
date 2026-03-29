// Probe: warp intrinsics in complex patterns — shuffle in loops,
// ballot for masking, vote in conditional chains, xor shuffle patterns.

// ------------------------------------------------------------------
// Warp reduction with __shfl_xor_sync (butterfly reduction).

__global__ void warp_xor_reduce(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? data[tid] : 0;
    // Butterfly reduction using xor shuffle
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    if (tid == 0) out[0] = val;
}

// ------------------------------------------------------------------
// Warp scan using __shfl_up_sync.

__global__ void warp_scan(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? data[tid] : 0;
    // Inclusive scan
    int lane = tid & 31;
    int tmp;
    tmp = __shfl_up_sync(0xFFFFFFFF, val, 1);
    if (lane >= 1)  val += tmp;
    tmp = __shfl_up_sync(0xFFFFFFFF, val, 2);
    if (lane >= 2)  val += tmp;
    tmp = __shfl_up_sync(0xFFFFFFFF, val, 4);
    if (lane >= 4)  val += tmp;
    tmp = __shfl_up_sync(0xFFFFFFFF, val, 8);
    if (lane >= 8)  val += tmp;
    tmp = __shfl_up_sync(0xFFFFFFFF, val, 16);
    if (lane >= 16) val += tmp;
    if (tid < n) out[tid] = val;
}

// ------------------------------------------------------------------
// __ballot_sync to count active lanes.

__global__ void ballot_count(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? data[tid] : 0;
    // Count how many lanes have v > 0
    unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
    // __popc-like count using bit manipulation
    unsigned int cnt = mask;
    cnt = cnt - ((cnt >> 1) & 0x55555555u);
    cnt = (cnt & 0x33333333u) + ((cnt >> 2) & 0x33333333u);
    cnt = (cnt + (cnt >> 4)) & 0x0F0F0F0Fu;
    cnt = cnt * 0x01010101u;
    cnt = cnt >> 24;
    if (tid == 0) out[0] = (int)cnt;
}

// ------------------------------------------------------------------
// Warp shuffle to broadcast from lane 0.

__global__ void warp_broadcast(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int val = (tid == 0) ? data[0] : 0;
    // Lane 0 broadcasts to all lanes
    val = __shfl_sync(0xFFFFFFFF, val, 0);
    if (tid < n) out[tid] = val;
}

// ------------------------------------------------------------------
// Shuffle in a loop: ring-shift values.

__global__ void warp_ring_shift(int *out, int *data, int n, int steps) {
    int tid = threadIdx.x;
    int val = (tid < n) ? data[tid] : 0;
    for (int s = 0; s < steps; s++) {
        val = __shfl_up_sync(0xFFFFFFFF, val, 1);
    }
    if (tid < n) out[tid] = val;
}

// ------------------------------------------------------------------
// Warp vote in loop condition: stop when no thread has positive val.

__global__ void warp_vote_loop(int *out, int *data, int n, int rounds) {
    int tid = threadIdx.x;
    int val = (tid < n) ? data[tid] : 0;
    int r = 0;
    for (int i = 0; i < rounds && __any_sync(0xFFFFFFFF, val > 0); i++) {
        if (val > 0) val--;
        r++;
    }
    if (tid < n) out[tid] = r;
}

// ------------------------------------------------------------------
// Float shuffle: broadcast min from lane 0 after reduction.

__global__ void float_warp_min(float *out, float *data, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? data[tid] : 3.40282e+38f;
    // Reduce to min
    float tmp;
    tmp = __shfl_down_sync(0xFFFFFFFF, val, 16);
    if (tmp < val) val = tmp;
    tmp = __shfl_down_sync(0xFFFFFFFF, val, 8);
    if (tmp < val) val = tmp;
    tmp = __shfl_down_sync(0xFFFFFFFF, val, 4);
    if (tmp < val) val = tmp;
    tmp = __shfl_down_sync(0xFFFFFFFF, val, 2);
    if (tmp < val) val = tmp;
    tmp = __shfl_down_sync(0xFFFFFFFF, val, 1);
    if (tmp < val) val = tmp;
    // Broadcast result from lane 0
    val = __shfl_sync(0xFFFFFFFF, val, 0);
    if (tid < n) out[tid] = val;
}
