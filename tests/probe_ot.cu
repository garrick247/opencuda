// Probe: complex real-world patterns — prefix scan, matrix transpose,
// warp-level voting operations, and multidimensional block indexing.

// ------------------------------------------------------------------
// Prefix sum (inclusive scan) within a warp using shfl_up_sync.
// Classic warp-level prefix scan algorithm.

__global__ void warp_prefix_scan(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        int v = data[tid];
        for (int delta = 1; delta < 32; delta <<= 1) {
            int prev = __shfl_up_sync(0xffffffff, v, delta);
            if ((tid & 31) >= delta) v += prev;
        }
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Matrix transpose: reads from row-major, writes column-major.
// Tests 2D index arithmetic: out[j*rows + i] = in[i*cols + j].

__global__ void matrix_transpose(float *out, float *in, int rows, int cols) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tile = 16;
    int row = by * tile + ty;
    int col = bx * tile + tx;
    if (row < rows && col < cols) {
        int in_idx  = row * cols + col;
        int out_idx = col * rows + row;
        out[out_idx] = in[in_idx];
    }
}

// ------------------------------------------------------------------
// Warp voting: __all_sync, __any_sync.

__global__ void warp_vote(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int all_positive = __all_sync(0xffffffff, v > 0);
        int any_negative = __any_sync(0xffffffff, v < 0);
        out[tid*2+0] = all_positive;
        out[tid*2+1] = any_negative;
    }
}

// ------------------------------------------------------------------
// 3D block indexing: threadIdx.z, blockIdx.z, blockDim.z.

__global__ void block_3d(int *out, int nx, int ny, int nz) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int bdz = blockDim.z;
    int gx = bx * bdx + tx;
    int gy = by * bdy + ty;
    int gz = bz * bdz + tz;
    if (gx < nx && gy < ny && gz < nz) {
        int idx = gz * ny * nx + gy * nx + gx;
        out[idx] = gx + gy * nx + gz * nx * ny;
    }
}
