// Stress: patterns that exercise every codegen path at once.
// Each kernel is designed to catch a specific class of silent bug.

// --- Nested loops with accumulator (tests loop writeback correctness) ---
__global__ void nested_loop_sum(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int s = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                s += a[(gid + i * 8 + j) % n];
            }
        }
        out[gid] = s;
    }
}

// --- Conditional in loop (tests phi node correctness) ---
__global__ void cond_in_loop(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int pos_sum = 0, neg_sum = 0;
        for (int i = 0; i < 16; i++) {
            int v = a[(gid + i) % n];
            if (v > 0) pos_sum += v;
            else neg_sum += v;
        }
        out[gid] = pos_sum + neg_sum;
    }
}

// --- Multiple device function calls in expression ---
__device__ int dev_add(int x, int y) { return x + y; }
__device__ int dev_mul(int x, int y) { return x * y; }

__global__ void multi_devfn(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = dev_add(dev_mul(a[gid], 3), dev_mul(b[gid], 7));
    }
}

// --- Shared mem + warp reduce + atomic (full pipeline) ---
__global__ void full_reduce(float *out, float *a, float *b, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;

    float v = (gid < n) ? a[gid] * a[gid] : 0.0f;  // square each

    // Warp reduce
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);

    if (lane == 0) smem[tid >> 5] = v;
    __syncthreads();

    // First warp reduces warp sums
    if (tid < 8) {
        v = smem[tid];
        v += __shfl_xor_sync(0xFF, v, 4);
        v += __shfl_xor_sync(0xFF, v, 2);
        v += __shfl_xor_sync(0xFF, v, 1);
        if (tid == 0) atomicAdd(out, v);
    }
}

// --- Type conversion chain: int → float → int (tests cvt correctness) ---
__global__ void cvt_chain(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int iv = a[gid];
        float fv = (float)iv * 0.7f + 0.5f;
        int back = (int)fv;
        out[gid] = back;
    }
}

// --- Shift + mask (tests bitwise codegen) ---
__global__ void pack_unpack(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int lo = a[gid] & 0xFFFF;
        int hi = b[gid] & 0xFFFF;
        int packed = (hi << 16) | lo;
        // Unpack and verify
        int lo2 = packed & 0xFFFF;
        int hi2 = (packed >> 16) & 0xFFFF;
        out[gid] = lo2 + hi2;  // should equal (a&0xFFFF) + (b&0xFFFF)
    }
}

// --- Ternary chain in loop (complex control flow) ---
__global__ void ternary_loop(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int score = 0;
        for (int i = 0; i < 8; i++) {
            int v = a[(gid + i) % n];
            score += (v > 50) ? 3 : (v > 0) ? 1 : (v > -50) ? -1 : -3;
        }
        out[gid] = score;
    }
}

// --- Warp prefix sum (specifically tests shfl.up clamp fix) ---
__global__ void warp_prefix(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = (gid < n) ? a[gid] : 0;
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if (lane >= d) v += t;
    }
    if (gid < n) out[gid] = v;
}
