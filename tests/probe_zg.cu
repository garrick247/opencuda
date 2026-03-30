// Probe: more real-world patterns — bitonic sort network, parallel prefix
// with bank-conflict-free shared memory, warp-cooperative reduction with
// final warp optimization, Conway's Game of Life stencil, parallel
// reduction with sequential addressing, and thread coarsening.

// ------------------------------------------------------------------
// Bitonic sort step (single kernel for one phase).

__global__ void bitonic_step(int *data, int j, int k, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int ixj = tid ^ j;
    if (ixj > tid) {
        if ((tid & k) == 0) {
            // Ascending
            if (data[tid] > data[ixj]) {
                int tmp = data[tid];
                data[tid] = data[ixj];
                data[ixj] = tmp;
            }
        } else {
            // Descending
            if (data[tid] < data[ixj]) {
                int tmp = data[tid];
                data[tid] = data[ixj];
                data[ixj] = tmp;
            }
        }
    }
}

// ------------------------------------------------------------------
// Parallel reduction with sequential addressing (no bank conflicts).

__global__ void reduce_seq_addr(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + tid;
    // Load two elements per thread
    float val = (gid < n) ? in[gid] : 0.0f;
    if (gid + blockDim.x < n) val += in[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();
    // Sequential addressing reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // Final warp (no syncthreads needed)
    if (tid < 32) {
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid +  8];
        smem[tid] += smem[tid +  4];
        smem[tid] += smem[tid +  2];
        smem[tid] += smem[tid +  1];
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Conway's Game of Life step.

__global__ void game_of_life(int *next, int *curr, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Count live neighbors (8-connected)
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                count += curr[ny * W + nx];
            }
        }
    }
    int alive = curr[y * W + x];
    // Rules: alive with 2-3 neighbors survives; dead with 3 neighbors born
    int next_state = (alive && (count == 2 || count == 3)) ||
                     (!alive && count == 3);
    next[y * W + x] = next_state ? 1 : 0;
}

// ------------------------------------------------------------------
// Thread coarsening: each thread processes COARSEN_FACTOR elements.

#define COARSEN 4

__global__ void coarsened_saxpy(float *y, float alpha, float *x, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    for (int i = 0; i < COARSEN; i++) {
        int idx = base + i;
        if (idx < n) y[idx] = alpha * x[idx] + y[idx];
    }
}

// ------------------------------------------------------------------
// Exclusive prefix sum (Blelloch scan) — up-sweep + down-sweep.

__global__ void blelloch_scan(int *data, int n) {
    __shared__ int temp[512];
    int tid = threadIdx.x;
    int offset = 1;

    // Load into shared memory
    temp[2*tid]   = (2*tid   < n) ? data[2*tid]   : 0;
    temp[2*tid+1] = (2*tid+1 < n) ? data[2*tid+1] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    for (int d = 256; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Set last to zero for exclusive scan
    if (tid == 0) temp[511] = 0;
    __syncthreads();

    // Down-sweep
    for (int d = 1; d < 512; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write back
    if (2*tid   < n) data[2*tid]   = temp[2*tid];
    if (2*tid+1 < n) data[2*tid+1] = temp[2*tid+1];
}

// ------------------------------------------------------------------
// Matrix-vector multiply (simple row-per-thread).

__global__ void matvec(float *y, float *A, float *x, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float s = 0.0f;
        for (int j = 0; j < N; j++) {
            s += A[row * N + j] * x[j];
        }
        y[row] = s;
    }
}
