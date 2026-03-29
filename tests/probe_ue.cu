// Probe: cooperative patterns, warp-level reductions with masks,
// multi-warp synchronization, and complex shared memory patterns.

// ------------------------------------------------------------------
// Warp-level exclusive prefix sum (scan within warp).

__global__ void warp_exclusive_scan(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        unsigned mask = __activemask();
        // Inclusive prefix sum using shuffle
        for (int offset = 1; offset < 32; offset *= 2) {
            int neighbor = __shfl_up_sync(mask, val, offset);
            if ((tid & 31) >= offset) {
                val += neighbor;
            }
        }
        // Convert to exclusive: shift right by 1 lane
        int excl = __shfl_up_sync(mask, val, 1);
        if ((tid & 31) == 0) excl = 0;
        else excl = val - in[tid];
        out[tid] = excl;
    }
}

// ------------------------------------------------------------------
// Block-level max reduction using shared memory tree.

__global__ void block_max_reduce(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : -2147483648;  // INT_MIN
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int a = smem[tid];
            int b = smem[tid + stride];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}

// ------------------------------------------------------------------
// Strided copy with coalescing (stride-1 read, stride-S write).

__global__ void strided_copy(float *out, float *in, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid * stride] = in[tid];
    }
}

// ------------------------------------------------------------------
// Warp vote + conditional count.

__global__ void warp_vote_count(int *out, int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned mask = 0xFFFFFFFF;
        // Count how many lanes have positive values
        int pos_count = __syncwarp_count(v > 0);
        // Check if all lanes have values >= 0
        int all_nonneg = __all_sync(mask, v >= 0);
        out[tid] = pos_count * 1000 + all_nonneg;
    }
}

// ------------------------------------------------------------------
// Tiled dot product (register-tiled inner loop).

__global__ void tile_dot(float *out, float *a, float *b, int n, int tile) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float acc = 0.0f;
        int base = gid * tile;
        for (int i = 0; i < tile; i++) {
            acc += a[base + i] * b[base + i];
        }
        out[gid] = acc;
    }
}

// ------------------------------------------------------------------
// Double-buffered shared memory (ping-pong pattern).

__global__ void double_buffer(float *out, float *in, int n) {
    __shared__ float buf[2][128];
    int tid = threadIdx.x;
    int lane = tid & 127;

    // Load first chunk
    if (blockIdx.x * 256 + lane < n)
        buf[0][lane] = in[blockIdx.x * 256 + lane];
    if (blockIdx.x * 256 + 128 + lane < n)
        buf[1][lane] = in[blockIdx.x * 256 + 128 + lane];
    __syncthreads();

    // Process both halves and write out
    int ping = tid / 128;
    float v = buf[ping][lane];
    v = v * v + 1.0f;
    if (blockIdx.x * 256 + tid < n)
        out[blockIdx.x * 256 + tid] = v;
}
