// Probe: multiple shared memory arrays, inter-block coordination,
// cooperative patterns, and bank-conflict-prone access patterns.

// ------------------------------------------------------------------
// Two separate shared memory arrays in one kernel.

__global__ void two_smem(float *out, float *a, float *b, int n) {
    __shared__ float sa[256];
    __shared__ float sb[256];
    int tid = threadIdx.x;
    if (tid < n) {
        sa[tid] = a[tid];
        sb[tid] = b[tid];
    }
    __syncthreads();
    if (tid < n) {
        // Access cross-element (potential bank conflict access pattern)
        int rev = blockDim.x - 1 - tid;
        out[tid] = sa[rev] + sb[rev];
    }
}

// ------------------------------------------------------------------
// Shared memory with stride (every other element).

__global__ void strided_smem(int *out, int *in, int n) {
    __shared__ int smem[512];
    int tid = threadIdx.x;
    // Write to even slots
    smem[tid * 2] = (tid < n) ? in[tid] : 0;
    smem[tid * 2 + 1] = (tid < n) ? in[tid] * 2 : 0;
    __syncthreads();
    if (tid < n) {
        out[tid] = smem[tid * 2] + smem[tid * 2 + 1];
    }
}

// ------------------------------------------------------------------
// Double-buffered shared memory pattern.

__global__ void double_buffer(float *out, float *in, int n) {
    __shared__ float buf[2][128];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load first tile into buf[0]
    int gid0 = bid * 128 + tid;
    buf[0][tid] = (gid0 < n) ? in[gid0] : 0.0f;
    __syncthreads();

    // Process buf[0], load next tile into buf[1]
    int gid1 = (bid + 1) * 128 + tid;  // next block's data
    buf[1][tid] = (gid1 < n) ? in[gid1] : 0.0f;
    __syncthreads();

    // Sum both buffers
    float r = buf[0][tid] + buf[1][tid];
    if (gid0 < n) out[gid0] = r;
}

// ------------------------------------------------------------------
// Shared memory reduction: log2 steps.

__global__ void smem_reduce(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 1; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Warp-level max reduction using shuffle.

__global__ void warp_max(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? in[gid] : -1e30f;
    unsigned mask = 0xFFFFFFFFu;
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 8));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 4));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out, v);  // not truly max but tests the pattern
    }
}

// ------------------------------------------------------------------
// Shared + global atomics combined.

__global__ void smem_global_atomic(int *out, int *in, int n) {
    __shared__ int local_hist[16];
    int tid = threadIdx.x;
    if (tid < 16) local_hist[tid] = 0;
    __syncthreads();

    if (tid < n) {
        int bin = in[tid] & 15;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    if (tid < 16) {
        atomicAdd(&out[tid], local_hist[tid]);
    }
}

// ------------------------------------------------------------------
// Butterfly transpose in shared memory.

__global__ void smem_transpose(float *out, float *in, int n) {
    __shared__ float tile[16][17];  // 17 wide to avoid bank conflicts
    int tid = threadIdx.x;
    int row = tid / 16;
    int col = tid % 16;
    int gid = blockIdx.x * 256 + tid;
    if (gid < n) tile[row][col] = in[gid];
    __syncthreads();
    // Read transposed
    float v = tile[col][row];
    if (gid < n) out[gid] = v;
}
