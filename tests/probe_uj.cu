// Probe: adversarial combinations — device function that modifies
// a struct via pointer, returns a value, AND has early exit;
// kernel with shared memory + atomics + warp ops in same block;
// deeply nested control flow with multiple mutation sites.

struct Stat {
    float sum;
    float sum_sq;
    int count;
};

// ------------------------------------------------------------------
// Device function: update stat via pointer, return flag.

__device__ int update_stat(Stat *s, float val, float threshold) {
    if (val < 0.0f) return -1;  // reject negatives
    s->sum    += val;
    s->sum_sq += val * val;
    s->count  += 1;
    return (val > threshold) ? 1 : 0;
}

// ------------------------------------------------------------------
// Kernel: accumulate stats + count threshold crossings.

__global__ void stat_accum(float *out, float *in, int n, float threshold) {
    int tid = threadIdx.x;
    if (tid < n) {
        Stat s;
        s.sum = 0.0f; s.sum_sq = 0.0f; s.count = 0;
        int crossings = 0;
        for (int i = tid; i < n; i += blockDim.x) {
            int flag = update_stat(&s, in[i], threshold);
            if (flag > 0) crossings++;
        }
        float mean = (s.count > 0) ? (s.sum / (float)s.count) : 0.0f;
        out[tid * 3]     = mean;
        out[tid * 3 + 1] = s.sum_sq;
        out[tid * 3 + 2] = (float)crossings;
    }
}

// ------------------------------------------------------------------
// Kernel: warp scan + shared prefix sum + global atomic.

__global__ void warp_shared_atomic(int *global_out, int *in, int n) {
    __shared__ int block_sums[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int warp = tid >> 5;

    int val = (gid < n) ? in[gid] : 0;

    // Warp-level inclusive scan
    unsigned mask = __activemask();
    for (int off = 1; off < 32; off <<= 1) {
        int nb = __shfl_up_sync(mask, val, off);
        if (lane >= off) val += nb;
    }

    // Write warp sum to shared
    if (lane == 31) block_sums[warp] = val;
    __syncthreads();

    // First warp scans block sums
    if (warp == 0) {
        int ws = (lane < (blockDim.x >> 5)) ? block_sums[lane] : 0;
        for (int off = 1; off < 32; off <<= 1) {
            int nb = __shfl_up_sync(0xFFFFFFFF, ws, off);
            if (lane >= off) ws += nb;
        }
        block_sums[lane] = ws;
    }
    __syncthreads();

    // Add prefix from previous warps
    int prefix = (warp > 0) ? block_sums[warp - 1] : 0;
    val += prefix;

    if (gid < n) {
        atomicAdd(global_out, val);
    }
}

// ------------------------------------------------------------------
// Deeply nested: triple loop with break/continue at each level.

__global__ void triple_loop_control(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        int v = in[tid];
        for (int i = 0; i < 4; i++) {
            if (i == 2 && v < 0) continue;  // skip i=2 for negatives
            for (int j = 0; j < 4; j++) {
                if (j > i) break;            // triangular iteration
                for (int k = 0; k < 4; k++) {
                    if (k == j) continue;    // skip diagonal
                    acc += i * j + k;
                }
            }
        }
        out[tid] = acc + (v > 0 ? 1 : -1);
    }
}

// ------------------------------------------------------------------
// Mixed types in conditional assignment chain.

__global__ void mixed_type_chain(float *out, int *flags, double *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int flag = flags[tid];
        double dv = vals[tid];
        float fv;
        // Chain of type conversions through conditionals
        if (flag == 0) {
            fv = (float)dv;
        } else if (flag == 1) {
            fv = (float)(dv * 2.0);
        } else if (flag == 2) {
            fv = (dv > 0.0) ? sqrtf((float)dv) : 0.0f;
        } else {
            fv = (float)((int)dv % 1000);
        }
        out[tid] = fv;
    }
}
