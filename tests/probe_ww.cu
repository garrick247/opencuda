// Probe: remaining atomic ops (atomicXor/Or/And/Inc/Dec),
// __ballot_sync patterns, shared memory bank conflicts avoidance,
// warp-level primitives, and conditional stores.

// ------------------------------------------------------------------
// atomicXor, atomicOr, atomicAnd on shared and global.

__global__ void atomic_bitwise(unsigned int *g_xor, unsigned int *g_or,
                                unsigned int *g_and, unsigned int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        atomicXor(g_xor, v);
        atomicOr(g_or,   v);
        atomicAnd(g_and, v);
    }
}

// ------------------------------------------------------------------
// atomicInc / atomicDec (wrapping increment/decrement).

__global__ void atomic_inc_dec(unsigned int *counter, unsigned int wrap, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicInc(counter, wrap);   // wraps at wrap
    }
}

// ------------------------------------------------------------------
// __ballot_sync to find the lowest active lane with a condition.

__global__ void ballot_find_first(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int cond = (gid < n) && (in[gid] > 0);
    unsigned int mask = __ballot_sync(0xFFFFFFFF, cond);
    // __ffs returns 1-based position of lowest set bit
    int first = __ffs(mask) - 1;  // convert to 0-based lane
    if (gid < n) {
        out[gid] = first;
    }
}

// ------------------------------------------------------------------
// Shared memory bank conflict avoidance: pad by 1 element.

__global__ void smem_padded(int *out, int *in, int n) {
    __shared__ int smem[32 + 1];   // +1 to avoid bank conflicts on transpose
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = in[gid];
    __syncthreads();

    // Each thread reads from a different bank row
    int partner = (tid + 16) % 32;
    if (gid < n) {
        out[gid] = smem[tid] + smem[partner];
    }
}

// ------------------------------------------------------------------
// Warp vote: count threads satisfying condition, then broadcast.

__global__ void warp_vote_count(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int active_count = __syncthreads_count(v > 0);
        out[gid] = active_count;
    }
}

// ------------------------------------------------------------------
// Conditional store: only write if changed.

__global__ void conditional_store(int *out, int *in, int *mask, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (mask[tid]) {
            out[tid] = in[tid];
        }
        // else: don't touch out[tid]
    }
}

// ------------------------------------------------------------------
// Double-buffered shared memory: ping-pong between two shared arrays.

__global__ void double_buffer(float *out, float *in, int n) {
    __shared__ float buf[2][128];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 128 + tid;

    // Load first chunk into buf[0]
    buf[0][tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Process buf[0] → buf[1]
    buf[1][tid] = buf[0][tid] * 2.0f;
    __syncthreads();

    if (gid < n) {
        out[gid] = buf[1][tid] + buf[0][tid];
    }
}

// ------------------------------------------------------------------
// Grid-stride loop pattern.

__global__ void grid_stride(int *out, int *in, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = in[i] * in[i];
    }
}

// ------------------------------------------------------------------
// Reduction using only warp shuffles, no shared memory.

__global__ void warp_sum_noshmem(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = (gid < n) ? in[gid] : 0;

    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);

    int lane = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    if (lane == 0) {
        atomicAdd(&out[warpid], v);
    }
}

// ------------------------------------------------------------------
// atomicExch: swap global values across threads.

__global__ void atomic_exch_swap(int *arr, int *new_vals, int *old_vals, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        old_vals[tid] = atomicExch(&arr[tid], new_vals[tid]);
    }
}
