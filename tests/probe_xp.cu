// Probe: __dp4a, __reduce_*_sync, __match_*_sync,
// atomicAdd on double, clock/clock64, __ldcg/__ldcs,
// and shfl on double.

// ------------------------------------------------------------------
// __dp4a: byte dot-product accumulate (SM_61+).

__global__ void dp4a_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // __dp4a(a, b, c) = byte[0]*byte[0] + ... + byte[3]*byte[3] + c
        int acc = 0;
        acc = __dp4a(a[tid], b[tid], acc);
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// __reduce_add_sync / __reduce_min_sync / __reduce_max_sync.

__global__ void warp_reduce_ops(int *out_add, int *out_min, int *out_max,
                                  int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = (gid < n) ? in[gid] : 0;
    unsigned int mask = __activemask();

    int s = __reduce_add_sync(mask, v);
    int mn = __reduce_min_sync(mask, v);
    int mx = __reduce_max_sync(mask, v);

    if ((threadIdx.x & 31) == 0 && gid < n) {
        out_add[gid / 32] = s;
        out_min[gid / 32] = mn;
        out_max[gid / 32] = mx;
    }
}

// ------------------------------------------------------------------
// __reduce_and_sync / __reduce_or_sync (unsigned variants).

__global__ void warp_reduce_bits(unsigned *out_and, unsigned *out_or,
                                   unsigned *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = (gid < n) ? in[gid] : 0xFFFFFFFF;
    unsigned int mask = __activemask();

    unsigned ra = __reduce_and_sync(mask, v);
    unsigned ro = __reduce_or_sync(mask, v);

    if ((threadIdx.x & 31) == 0 && gid < n) {
        out_and[gid / 32] = ra;
        out_or[gid / 32]  = ro;
    }
}

// ------------------------------------------------------------------
// __match_any_sync: which threads in mask share the same value.

__global__ void match_any_kernel(unsigned *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        unsigned int mask = __activemask();
        // Returns bitmask of threads in mask that have same value as caller
        unsigned hits = __match_any_sync(mask, in[gid]);
        out[gid] = hits;
    }
}

// ------------------------------------------------------------------
// __match_all_sync: check if all threads in mask share the same value.

__global__ void match_all_kernel(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        unsigned int mask = __activemask();
        int pred;
        unsigned hits = __match_all_sync(mask, in[gid], &pred);
        out[gid] = pred;
        (void)hits;
    }
}

// ------------------------------------------------------------------
// atomicAdd on double.

__global__ void atomic_add_double(double *sum, double *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicAdd(sum, in[gid]);
    }
}

// ------------------------------------------------------------------
// clock() and clock64() as timing markers.

__global__ void timing_kernel(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    long long t0 = clock64();
    int s = 0;
    for (int i = 0; i < n; i++) s += in[i];
    long long t1 = clock64();
    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = (long long)clock();
        out[2] = (long long)s;  // prevent dead-code elim
    }
}

// ------------------------------------------------------------------
// __ldcg / __ldcs (cache-evict / streaming loads).

__global__ void ldcg_ldcs_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float va = __ldcg(&a[tid]);
        float vb = __ldcs(&b[tid]);
        out[tid] = va + vb;
    }
}

// ------------------------------------------------------------------
// shfl on float (verify the b32 path for float correctly uses %f regs).

__global__ void shfl_float_bcast(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = in[gid];
        // Broadcast lane 0 of each warp's value to all lanes
        float bcast = __shfl_sync(0xFFFFFFFF, v, 0);
        out[gid] = bcast;
    }
}
