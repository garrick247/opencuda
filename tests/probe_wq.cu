// Probe: __activemask(), clock64() timing, atomicAdd float return,
// shfl_sync float, warp reduction returning float, and
// cooperative global memory patterns.

// ------------------------------------------------------------------
// __activemask(): returns bitmask of active lanes.

__global__ void activemask_kernel(unsigned int *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        unsigned int mask = __activemask();
        out[gid] = mask;
    }
}

// ------------------------------------------------------------------
// clock64() for timing measurement.

__global__ void clock_kernel(long long *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        long long start = clock64();
        // Dummy computation to prevent optimization
        int v = threadIdx.x;
        for (int i = 0; i < 8; i++) v = v * 3 + 1;
        long long end = clock64();
        out[gid] = end - start + (long long)v;
    }
}

// ------------------------------------------------------------------
// atomicAdd float: return value is the OLD value.

__global__ void atomic_add_float_return(float *out, float *acc, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float old = atomicAdd(acc, in[gid]);
        out[gid] = old;   // old value before addition
    }
}

// ------------------------------------------------------------------
// __shfl_sync with float: type preserved (float shuffle).

__global__ void shfl_float(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = in[gid];
        // Each thread gets lane 0's float value
        float lane0 = __shfl_sync(0xFFFFFFFF, v, 0);
        out[gid] = lane0;
    }
}

// ------------------------------------------------------------------
// Warp reduction of floats using shfl_xor.

__global__ void warp_reduce_float(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? in[gid] : 0.0f;

    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);

    if ((threadIdx.x & 31) == 0 && gid < n) {
        out[gid / 32] = v;
    }
}

// ------------------------------------------------------------------
// __trap() — used in asserts.

__global__ void trap_on_error(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v < 0) {
            __trap();   // terminate if negative
        }
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Mixed atomicMin on shared and global memory.

__global__ void atomic_shared_global(int *gmin, int *in, int n) {
    __shared__ int smin[1];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (tid == 0) smin[0] = 2147483647;  // INT_MAX
    __syncthreads();

    if (gid < n) {
        atomicMin(&smin[0], in[gid]);   // block-local min
    }
    __syncthreads();

    if (tid == 0) {
        atomicMin(gmin, smin[0]);       // global min
    }
}

// ------------------------------------------------------------------
// Cooperative pattern: each thread writes to shared, then all read.

__global__ void cooperative_shared(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;

    // Phase 1: all write
    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Phase 2: each reads offset by blockDim.x/2 (wrapping)
    int partner = (tid + blockDim.x / 2) % blockDim.x;
    if (gid < n) {
        out[gid] = smem[tid] + smem[partner];
    }
}
