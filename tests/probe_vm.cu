// Probe: bit manipulation intrinsics (__popc, __clz, __ffs, __brev),
// shared memory array of structs, and warp match intrinsics.

// ------------------------------------------------------------------
// Population count.

__global__ void popc_ops(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        int cnt32 = __popc(v);            // 32-bit popcount
        int cnt64 = __popcll((unsigned long long)v * v);  // 64-bit popcount
        out[tid] = cnt32 + (int)cnt64;
    }
}

// ------------------------------------------------------------------
// Count leading zeros.

__global__ void clz_ops(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        int clz32 = __clz(v);              // count leading zeros in 32-bit
        int clz64 = __clzll((unsigned long long)v);  // 64-bit
        out[tid] = clz32 + (int)clz64;
    }
}

// ------------------------------------------------------------------
// Find first set bit.

__global__ void ffs_ops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = __ffs(v);   // 1-indexed position of LSB, or 0 if v==0
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Bit reversal.

__global__ void brev_ops(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int r = __brev(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Combined bit ops: ffs + clz + popc in one kernel.

__global__ void bit_combined(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        int p = __popc(v);
        int z = __clz(v);
        int f = __ffs((int)v);
        out[tid] = p + z + f;
    }
}

// ------------------------------------------------------------------
// Shared memory of structs.

struct Elem {
    float val;
    int idx;
};

__global__ void shared_struct_arr(float *out, float *in, int *indices, int n) {
    __shared__ struct Elem smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load into shared struct array
    if (gid < n) {
        smem[tid].val = in[gid];
        smem[tid].idx = indices[gid];
    } else {
        smem[tid].val = 0.0f;
        smem[tid].idx = -1;
    }
    __syncthreads();

    // Each thread uses a neighbor's value
    if (gid < n) {
        int nb = smem[tid].idx & (blockDim.x - 1);  // neighbor index in block
        out[gid] = smem[tid].val + smem[nb].val;
    }
}

// ------------------------------------------------------------------
// __match_any_sync: find threads with same value.

__global__ void match_any(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        unsigned mask = __activemask();
        // returns a bitmask of lanes with the same value
        unsigned same = __match_any_sync(mask, (unsigned)v);
        out[gid] = __popc(same);  // count of threads with same value
    }
}
