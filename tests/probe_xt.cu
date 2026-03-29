// Probe: __sad/__usad, warp prefix sum via ballot, complex shfl patterns,
// conditional atomic, loop-carried phi with bool, and global function recursion sim.

// ------------------------------------------------------------------
// __sad and __usad (sum of absolute differences).

__global__ void sad_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        acc = __sad(a[tid], b[tid], acc);
        out[tid] = acc;
    }
}

__global__ void usad_kernel(unsigned *out, unsigned *a, unsigned *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned acc = 0;
        acc = __usad(a[tid], b[tid], acc);
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Warp prefix sum using __ballot_sync + __popc.

__global__ void warp_prefix_ballot(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    if (gid < n) {
        int v = in[gid];
        // Count how many lower-indexed active threads have value > 0
        unsigned mask = __ballot_sync(0xFFFFFFFF, v > 0);
        // Prefix: bits [0..lane-1]
        unsigned prefix_mask = mask & ((1u << lane) - 1u);
        out[gid] = __popc(prefix_mask);
    }
}

// ------------------------------------------------------------------
// Conditional atomic: only write if value exceeds threshold.

__global__ void cond_atomic(int *max_val, int *in, int threshold, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && in[gid] > threshold) {
        atomicMax(max_val, in[gid]);
    }
}

// ------------------------------------------------------------------
// Loop with bool loop-carried variable.

__global__ void bool_loop_carry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int found = 0;
        int pos = -1;
        for (int i = 0; i < 32 && !found; i++) {
            if ((v >> i) & 1) {
                found = 1;
                pos = i;
            }
        }
        out[tid] = pos;  // position of LSB (or -1 if zero)
    }
}

// ------------------------------------------------------------------
// Multiple shfl to broadcast + reduce in a single pass.

__global__ void shfl_multi_reduce(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? in[gid] : 0.0f;
    // Full warp reduce: sum
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if ((threadIdx.x & 31) == 0 && gid < n) out[gid / 32] = v;
}

// ------------------------------------------------------------------
// Prefix scan using __shfl_up_sync.

__global__ void shfl_prefix_scan(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int lane = threadIdx.x & 31;
        for (int offset = 1; offset < 32; offset <<= 1) {
            int y = __shfl_up_sync(0xFFFFFFFF, v, offset);
            if (lane >= offset) v += y;
        }
        out[gid] = v;
    }
}

// ------------------------------------------------------------------
// __mulhi and __mul24 arithmetic.

__global__ void mulhi_mul24(int *out_hi, int *out_24, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int hi = __mulhi(a[tid], b[tid]);
        int lo24 = __mul24(a[tid] & 0xFFFFFF, b[tid] & 0xFFFFFF);
        out_hi[tid]  = hi;
        out_24[tid]  = lo24;
    }
}

// ------------------------------------------------------------------
// __brev and __clz on edge-case values.

__global__ void brev_clz(unsigned *out_brev, int *out_clz, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned v = in[tid];
        out_brev[tid] = __brev(v);
        out_clz[tid]  = __clz((int)v);
    }
}
