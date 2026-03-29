// Probe: 64-bit atomics, warp shuffle on float/int,
// __ballot_sync, __activemask, lane masking,
// warp reduction pattern

// 64-bit atomic add (float)
__global__ void atomic64_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(out, in[tid]);
    }
}

// 64-bit atomic add (int)
__global__ void atomic64_int(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(out, in[tid]);
    }
}

// atomicMin / atomicMax on shared memory
__global__ void atomic_shared_minmax(int *out, int *in, int n) {
    __shared__ int shared_min;
    __shared__ int shared_max;
    int tid = threadIdx.x;
    if (tid == 0) {
        shared_min = in[0];
        shared_max = in[0];
    }
    __syncthreads();
    if (tid < n) {
        atomicMin(&shared_min, in[tid]);
        atomicMax(&shared_max, in[tid]);
    }
    __syncthreads();
    if (tid == 0) {
        out[0] = shared_min;
        out[1] = shared_max;
    }
}

// Warp shuffle down — float version
__global__ void warp_shfl_down_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    // Reduce within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (tid == 0) out[0] = val;
}

// __ballot_sync and popcount of ballot
__global__ void ballot_popcount(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
    if (tid == 0) {
        // Count bits set (popcount)
        unsigned int cnt = 0;
        unsigned int m = mask;
        while (m) {
            cnt += m & 1;
            m >>= 1;
        }
        out[0] = (int)cnt;
    }
}
