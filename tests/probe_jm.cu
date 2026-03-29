// Probe: atomics with computed addresses and loop,
// warp-level reduction using __shfl_down_sync,
// multi-dimensional shared memory with computed row/col indices,
// __ballot_sync result used in arithmetic

// Atomic histogram with computed bucket index
__global__ void atomic_histogram(int *hist, int *in, int n, int nbuckets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int bucket = v % nbuckets;
        if (bucket >= 0 && bucket < nbuckets) {
            atomicAdd(&hist[bucket], 1);
        }
    }
}

// Warp reduction using shfl_down
__global__ void warp_sum(int *out, int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    // Reduce within warp using butterfly pattern
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(out, val);
    }
}

// 2D shared memory tile: each thread fills one cell, then reads a neighbor
__global__ void tile_shift(float *out, float *in, int width, int height) {
    __shared__ float tile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * 16 + tx;
    int gy = blockIdx.y * 16 + ty;
    if (gx < width && gy < height) {
        tile[ty][tx] = in[gy * width + gx];
    }
    __syncthreads();
    // Each thread reads the cell to its right (wrapping within tile)
    int rx = (tx + 1) % 16;
    if (gx < width && gy < height) {
        out[gy * width + gx] = tile[ty][rx];
    }
}

// Atomic compare-and-swap loop (spin until value changes)
__global__ void atomic_cas_update(int *lock, int *counter) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int old_val, new_val;
        do {
            old_val = *counter;
            new_val = old_val + 1;
        } while (atomicCAS(counter, old_val, new_val) != old_val);
    }
}
