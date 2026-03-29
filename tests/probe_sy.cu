// Probe: __clz/__popc/__ffs builtins, warpSize, ballot used as value,
// and multi-warp coordination patterns.

// ------------------------------------------------------------------
// __clz (count leading zeros).

__global__ void clz_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __clz(in[tid]);
    }
}

// ------------------------------------------------------------------
// __popc (popcount).

__global__ void popc_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __popcll((unsigned long long)in[tid]);
    }
}

// ------------------------------------------------------------------
// __ffs (find first set).

__global__ void ffs_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ffs(in[tid]);
    }
}

// ------------------------------------------------------------------
// Ballot sync as a bitmask fed into popcount.

__global__ void ballot_popc(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
        out[tid] = __popc(mask);
    }
}

// ------------------------------------------------------------------
// Warp leader reduction using shfl_down + ballot.

__global__ void warp_leader_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int val = (tid < n) ? in[tid] : 0;

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane == 0 && tid < n) {
        out[tid >> 5] = val;  // store warp sum
    }
}

// ------------------------------------------------------------------
// __clz used to compute floor(log2(x)).

__global__ void floor_log2(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid] = (v == 0) ? -1 : (31 - __clz(v));
    }
}

// ------------------------------------------------------------------
// Bit reversal using shifts.

__global__ void bit_reverse(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int r = 0;
        for (int i = 0; i < 32; i++) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Predicate mask built from multiple conditions.

__global__ void multi_cond_mask(unsigned int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned int m0 = __ballot_sync(0xFFFFFFFF, v > 0);
        unsigned int m1 = __ballot_sync(0xFFFFFFFF, v > 100);
        unsigned int m2 = __ballot_sync(0xFFFFFFFF, v < -100);
        out[tid] = (m0 & ~m1) | m2;
    }
}
