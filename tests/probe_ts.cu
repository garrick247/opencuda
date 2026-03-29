// Probe: activemask, syncthreads_count/and/or, __trap, and warpSize
// used in complex expressions.

// ------------------------------------------------------------------
// __activemask().

__global__ void activemask_use(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int mask = __activemask();
        out[tid] = __popc(mask);  // count active threads in warp
    }
}

// ------------------------------------------------------------------
// __syncthreads_count.

__global__ void sync_count(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    int count = __syncthreads_count(v > 0);  // count threads where v > 0
    if (tid < n) out[tid] = count;
}

// ------------------------------------------------------------------
// __syncthreads_and.

__global__ void sync_and(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 1;
    int all_positive = __syncthreads_and(v > 0);  // 1 if ALL threads have v > 0
    if (tid < n) out[tid] = all_positive;
}

// ------------------------------------------------------------------
// __syncthreads_or.

__global__ void sync_or(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    int any_positive = __syncthreads_or(v > 0);  // 1 if ANY thread has v > 0
    if (tid < n) out[tid] = any_positive;
}

// ------------------------------------------------------------------
// warpSize in complex expressions.

__global__ void warp_ops(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lane = tid & (warpSize - 1);
        int wid  = tid / warpSize;
        int half_warp = warpSize / 2;
        out[tid] = lane + wid * warpSize + half_warp;
    }
}

// ------------------------------------------------------------------
// Lane ID and warp ID using bitops.

__global__ void warp_partition(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lane = tid % warpSize;
        int is_even_lane = (lane % 2 == 0) ? 1 : 0;
        int is_first_in_warp = (lane == 0) ? 1 : 0;
        out[tid] = is_even_lane * 10 + is_first_in_warp;
    }
}

// ------------------------------------------------------------------
// __activemask used to count active threads before reduction.

__global__ void active_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    unsigned int active = __activemask();
    // Only active threads participate in shfl reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(active, val, offset);
    }
    if ((tid & (warpSize - 1)) == 0 && tid < n) {
        out[tid / warpSize] = val;
    }
}
