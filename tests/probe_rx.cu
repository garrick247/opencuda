// Probe: atomics stress — all variants, mixed-type atomic patterns,
// atomic on shared memory, and global histogram patterns.

// ------------------------------------------------------------------
// All atomic ops on global int.

__global__ void all_atomics(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        atomicAdd(out + 0, v);
        atomicSub(out + 1, v);
        atomicMin(out + 2, v);
        atomicMax(out + 3, v);
        atomicAnd(out + 4, v);
        atomicOr(out + 5, v);
        atomicXor(out + 6, v);
        atomicExch(out + 7, v);
    }
}

// ------------------------------------------------------------------
// Atomic on float: only atomicAdd is natively supported.

__global__ void atomic_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(out, in[tid]);
    }
}

// ------------------------------------------------------------------
// AtomicCAS: compare-and-swap.

__global__ void atomic_cas(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int old = atomicCAS(out, 0, in[tid]);
        // old is the previous value; ignore it here
        (void)old;
    }
}

// ------------------------------------------------------------------
// Atomic on shared memory histogram.

__global__ void shared_histogram(int *out, int *in, int n) {
    __shared__ int hist[16];
    int tid = threadIdx.x;
    if (tid < 16) hist[tid] = 0;
    __syncthreads();
    if (tid < n) {
        int bin = in[tid] & 15;
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();
    if (tid < 16) {
        atomicAdd(&out[tid], hist[tid]);
    }
}

// ------------------------------------------------------------------
// AtomicAdd on unsigned int.

__global__ void atomic_uint(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(out, in[tid]);
    }
}

// ------------------------------------------------------------------
// Atomic increment/decrement with bounds check pattern.

__global__ void bounded_inc(int *counter, int *out, int limit, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int old = atomicAdd(counter, 1);
        if (old < limit) {
            out[old] = tid;
        }
    }
}

// ------------------------------------------------------------------
// AtomicAdd on long long.

__global__ void atomic_ll(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(out, in[tid]);
    }
}

// ------------------------------------------------------------------
// Warp-level reduction then atomic for final sum.

__global__ void warp_reduce_atomic(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    // Simple warp reduction with __shfl_down_sync
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    if ((tid & 31) == 0) {
        atomicAdd(out, v);
    }
}
