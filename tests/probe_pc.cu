// Probe: shared memory with multiple arrays, conditional shared write,
// extern shared with cast, and atomics on shared memory.

// ------------------------------------------------------------------
// Multiple named __shared__ arrays in same kernel.

__global__ void multi_shared(float *out, float *in, int n) {
    __shared__ float smA[32];
    __shared__ float smB[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        smA[tid] = in[tid];
        smB[tid] = in[tid] * 2.0f;
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        out[tid] = smA[tid] + smB[tid];
    }
}

// ------------------------------------------------------------------
// Conditional write to shared memory: only some threads write.
// Others read the default value (0).

__global__ void cond_shared_write(int *out, int *flags, int *vals, int n) {
    __shared__ int sm[32];
    int tid = threadIdx.x;
    sm[tid] = 0;
    __syncthreads();
    if (tid < n && flags[tid]) {
        sm[tid] = vals[tid];
    }
    __syncthreads();
    out[tid] = sm[tid];
}

// ------------------------------------------------------------------
// Extern shared array cast to int: dynamic shared memory typed as int.

__global__ void extern_shared_int(int *out, int *in, int n) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = in[tid];
    }
    __syncthreads();
    if (tid < n) {
        out[tid] = smem[tid] + smem[(tid + 1) % n];
    }
}

// ------------------------------------------------------------------
// Shared memory used as scratch in warp scan.
// Each warp writes to its own section of shared memory.

__global__ void warp_shared_scan(int *out, int *data, int n) {
    __shared__ int sm[32];
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    if (tid < n) {
        sm[lane] = data[tid];
    }
    __syncthreads();
    // Simple prefix sum in shared mem for first warp
    if (warp == 0 && lane < n) {
        int v = sm[lane];
        for (int s = 1; s < 32; s <<= 1) {
            if (lane >= s) v += sm[lane - s];
            __syncthreads();
            sm[lane] = v;
            __syncthreads();
        }
        out[lane] = sm[lane];
    }
}

// ------------------------------------------------------------------
// atomicAdd on shared memory (not global).

__global__ void shared_atomic(int *out, int *data, int n) {
    __shared__ int counter;
    int tid = threadIdx.x;
    if (tid == 0) counter = 0;
    __syncthreads();
    if (tid < n && data[tid] > 0) {
        atomicAdd(&counter, 1);
    }
    __syncthreads();
    if (tid == 0) {
        out[0] = counter;
    }
}
