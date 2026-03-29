// Probe: complex __device__ fn control flow, 2D thread flat index,
// string literal in printf, and chained ternary depth.

// ------------------------------------------------------------------
// __device__ fn with two early returns and a final return.
// Tests that all three return paths are correctly inlined.

__device__ int clamp_signed(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__global__ void clamp_kernel(int *out, int *data, int n, int lo, int hi) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = clamp_signed(data[tid], lo, hi);
    }
}

// ------------------------------------------------------------------
// 2D thread flat index: row-major global ID from (bx, by, tx, ty).

__global__ void flat_index_2d(int *out, int width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int gx = bx * bdx + tx;
    int gy = by * bdy + ty;
    out[gy * width + gx] = gx + gy * width;
}

// ------------------------------------------------------------------
// Chained ternary 4 levels deep.
// Tests that nested ternary codegen correctly chains the merge blocks.

__global__ void deep_ternary(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r = (v < 0) ? -1 :
                (v == 0) ? 0 :
                (v < 10) ? 1 :
                (v < 100) ? 2 : 3;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Shared memory tiling: 2D tile loaded into __shared__, then read.
// Classic matrix-multiply-style shared memory access pattern.

__global__ void shared_tile(float *out, float *in, int rows, int cols) {
    __shared__ float tile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    if (row < rows && col < cols) {
        tile[ty][tx] = in[row * cols + col];
    }
    __syncthreads();
    if (row < rows && col < cols) {
        out[row * cols + col] = tile[ty][tx];
    }
}

// ------------------------------------------------------------------
// Loop with multiple exits: break on sentinel, also bounded by n.

__global__ void multi_exit_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] == -1) break;   // sentinel exit
            sum += data[i];
        }
        out[0] = sum;
    }
}
