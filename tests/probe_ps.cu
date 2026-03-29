// Probe: __shared__ memory patterns, __syncthreads placement,
// tiled algorithms, shared + register interaction.

// ------------------------------------------------------------------
// Shared memory tile load then sync then compute.
// Classic: load tile into smem, sync, reduce within tile.

__global__ void smem_tile_reduce(int *out, int *data, int n) {
    __shared__ int tile[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        tile[tid] = data[tid];
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n && i < 32; i++) {
            sum += tile[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Shared memory with multiple sync points.
// Load → sync → compute → store → sync → load computed result.

__global__ void smem_double_sync(int *out, int *data, int n) {
    __shared__ int a[32];
    __shared__ int b[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        a[tid] = data[tid];
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        b[tid] = a[tid] * 2 + a[(tid + 1) % 32];
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n && i < 32; i++) {
            sum += b[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// 1D convolution using shared memory.
// Each thread loads one element, then computes sum of 3 neighbors.

__global__ void smem_conv1d(int *out, int *data, int n) {
    __shared__ int smem[34];  // 32 + 2 halo
    int tid = threadIdx.x;
    if (tid < 32 && tid < n) {
        smem[tid + 1] = data[tid];
    }
    // Load halo
    if (tid == 0) {
        smem[0] = (n > 0) ? data[0] : 0;
    }
    if (tid == 31) {
        smem[33] = (n > 32) ? data[32] : 0;
    }
    __syncthreads();
    if (tid < 32 && tid < n) {
        out[tid] = smem[tid] + smem[tid + 1] + smem[tid + 2];
    }
}

// ------------------------------------------------------------------
// Prefix sum (scan) using shared memory.
// Each thread loads, then iterative doubling in smem.

__global__ void smem_scan(int *out, int *data, int n) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        smem[tid] = data[tid];
    } else if (tid < 32) {
        smem[tid] = 0;
    }
    __syncthreads();
    // Only thread 0 does sequential prefix sum for simplicity
    if (tid == 0) {
        for (int i = 1; i < 32 && i < n; i++) {
            smem[i] += smem[i - 1];
        }
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        out[tid] = smem[tid];
    }
}

// ------------------------------------------------------------------
// Shared memory used as scratch for multi-pass computation.
// Pass 1: load & compute partial results into smem.
// Pass 2: reduce smem into single output.

__global__ void smem_two_pass(int *out, int *a, int *b, int n) {
    __shared__ int scratch[32];
    int tid = threadIdx.x;
    // Pass 1: each thread computes a[tid] * b[tid]
    if (tid < 32 && tid < n) {
        scratch[tid] = a[tid] * b[tid];
    }
    __syncthreads();
    // Pass 2: thread 0 sums scratch
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n && i < 32; i++) {
            total += scratch[i];
        }
        out[0] = total;
    }
}

// ------------------------------------------------------------------
// Shared memory with conditional write.
// Some threads may not write to smem — others must read 0.

__global__ void smem_cond_write(int *out, int *data, int *mask, int n) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    // Initialize to 0 first
    if (tid < 32) {
        smem[tid] = 0;
    }
    __syncthreads();
    if (tid < n && tid < 32 && mask[tid]) {
        smem[tid] = data[tid];
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 32; i++) {
            sum += smem[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Warp shuffle for in-warp reduction (no shared memory needed).
// Tests __shfl_down_sync reduction.

__global__ void warp_reduce(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? data[tid] : 0;
    // Warp-level reduction using shuffle down
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    if (tid == 0) {
        out[0] = val;
    }
}

// ------------------------------------------------------------------
// Shared memory + warp shuffle combined.
// Warp reduce first, then lane 0 stores into shared memory.

__global__ void smem_warp_hybrid(int *out, int *data, int n) {
    __shared__ int per_warp[4];  // up to 4 warps
    int tid   = threadIdx.x;
    int warpid = tid / 32;
    int lane   = tid % 32;
    int val = (tid < n) ? data[tid] : 0;
    // Warp reduce
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    if (lane == 0 && warpid < 4) {
        per_warp[warpid] = val;
    }
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < 4; i++) {
            total += per_warp[i];
        }
        out[0] = total;
    }
}
