// Probe: warp intrinsics — vote, ballot, match, and prefix operations.

// ------------------------------------------------------------------
// __ballot_sync: returns bitmask of which lanes have pred=true.

__global__ void ballot_test(unsigned int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    unsigned int mask = __ballot_sync(0xFFFFFFFFu, v > 0);
    if (tid == 0) out[0] = mask;
}

// ------------------------------------------------------------------
// __any_sync: true if any lane in mask has pred=true.

__global__ void any_sync_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    int any_pos = __any_sync(0xFFFFFFFFu, v > 0);
    int any_neg = __any_sync(0xFFFFFFFFu, v < 0);
    if (tid == 0) {
        out[0] = any_pos;
        out[1] = any_neg;
    }
}

// ------------------------------------------------------------------
// __all_sync: true if all lanes in mask have pred=true.

__global__ void all_sync_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 1;
    int all_pos = __all_sync(0xFFFFFFFFu, v > 0);
    if (tid == 0) out[0] = all_pos;
}

// ------------------------------------------------------------------
// __shfl_sync: broadcast from lane 0.

__global__ void broadcast_sync(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    // Broadcast lane 0's value to all lanes
    int bcast = __shfl_sync(0xFFFFFFFFu, v, 0);
    if (tid < n) out[tid] = bcast;
}

// ------------------------------------------------------------------
// __shfl_up_sync: shift values upward (prefix sum style).

__global__ void shfl_up_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    int up1 = __shfl_up_sync(0xFFFFFFFFu, v, 1);
    int up2 = __shfl_up_sync(0xFFFFFFFFu, v, 2);
    if (tid < n) out[tid] = v + up1 + up2;
}

// ------------------------------------------------------------------
// __shfl_xor_sync: butterfly reduction.

__global__ void butterfly_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    if (tid < n) out[tid] = v;
}

// ------------------------------------------------------------------
// __shfl_down_sync: parallel prefix max.

__global__ void shfl_down_max(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float v = (tid < n) ? in[tid] : 0.0f;
    unsigned mask = 0xFFFFFFFFu;
    float d1 = __shfl_down_sync(mask, v, 1);
    float d2 = __shfl_down_sync(mask, v, 2);
    float d4 = __shfl_down_sync(mask, v, 4);
    float r = v;
    if (d1 > r) r = d1;
    if (d2 > r) r = d2;
    if (d4 > r) r = d4;
    if (tid < n) out[tid] = r;
}

// ------------------------------------------------------------------
// Warp-level popcount of ballot.

__global__ void warp_popcount(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    unsigned int ballot = __ballot_sync(0xFFFFFFFFu, v > 0);
    int count = __popc(ballot);
    if (tid < n) out[tid] = count;
}
