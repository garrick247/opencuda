// Probe: Warp intrinsics and cooperative patterns with complex expressions
// - __ballot_sync result used in branch condition
// - __activemask() + __ballot_sync combo
// - warp shuffle in while loop (reduction)
// - __any_sync / __all_sync
// - atomicCAS pattern
// - Multiple warp-level operations in sequence

__global__ void warp_vote_branch(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    // Vote: does any lane in the warp have val > 0?
    unsigned int mask = __ballot_sync(0xFFFFFFFF, val > 0.0f);
    if (mask != 0u) {
        // At least one lane has positive value
        if (tid < n) out[tid] = val * 2.0f;
    } else {
        if (tid < n) out[tid] = 0.0f;
    }
}

__global__ void warp_any_all(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    int any_pos = __any_sync(0xFFFFFFFF, val > 0);
    int all_pos = __all_sync(0xFFFFFFFF, val > 0);
    if (tid < n) {
        out[tid] = any_pos * 2 + all_pos;
    }
}

// Warp reduction via shuffle in a while loop
__global__ void warp_reduce_while(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    int offset = 16;
    while (offset > 0) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        offset >>= 1;
    }
    if ((tid & 31) == 0) {
        atomicAdd(out, val);
    }
}

// atomicCAS for spin-lock style
__global__ void atomic_cas_pattern(int *lock, int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // acquire lock
        int old;
        do {
            old = atomicCAS(lock, 0, 1);
        } while (old != 0);
        // critical section
        int sum = 0;
        for (int i = 0; i < n; i++) sum += in[i];
        out[0] = sum;
        // release lock
        atomicExch(lock, 0);
    }
}

// Multiple shuffle operations in same expression
__global__ void multi_shuffle(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    float val = (tid < n) ? in[tid] : 0.0f;
    // Broadcast lane 0 value to all lanes, and lane 31 to all
    float v0 = __shfl_sync(0xFFFFFFFF, val, 0);
    float v31 = __shfl_sync(0xFFFFFFFF, val, 31);
    if (tid < n) {
        out[tid] = val + v0 * 0.5f + v31 * 0.5f;
    }
}

// Ballot result used in arithmetic
__global__ void ballot_arithmetic(unsigned int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    unsigned int pos_mask = __ballot_sync(0xFFFFFFFF, val > 0);
    unsigned int neg_mask = __ballot_sync(0xFFFFFFFF, val < 0);
    if (tid < n) {
        // Number of lanes with positive / negative
        out[tid] = __popc(pos_mask) + __popc(neg_mask);
    }
}
