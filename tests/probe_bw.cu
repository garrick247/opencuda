// Probe: Multi-dimensional threadIdx/blockIdx patterns
// - 2D grid with Y and Z dimensions
// - Flat thread ID from 3D grid
// - Block stride loops (grid stride)

__global__ void grid_stride_loop(float *out, float *in, int n) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += stride) {
        out[i] = in[i] * 2.0f;
    }
}

__global__ void flat_3d_tid(int *out, int W, int H, int D) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x < W && y < H && z < D) {
        int flat = x + y * W + z * W * H;
        out[flat] = flat;
    }
}

__global__ void col_major_access(float *out, float *in, int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < rows && col < cols) {
        // Column-major: index = col * rows + row
        int col_idx = col * rows + row;
        // Row-major: index = row * cols + col
        int row_idx = row * cols + col;
        out[row_idx] = in[col_idx];
    }
}
