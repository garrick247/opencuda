// Probe: multi-dimensional array declarations, 2D array indexing,
// flat 2D access (arr[row*width+col])

__global__ void mat_transpose(float *out, float *in, int rows, int cols) {
    int tid = threadIdx.x;
    int row = tid / cols;
    int col = tid % cols;
    if (tid < rows * cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

// Shared memory 2D tile
__global__ void tiled_copy(float *out, float *in, int n) {
    __shared__ float tile[16 * 16];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * 256 + tid;
    if (gid < n) {
        tile[tid] = in[gid];
        __syncthreads();
        out[gid] = tile[tid];
    }
}

// Row-stride access pattern
__global__ void row_scale(float *mat, float *scales, int rows, int cols) {
    int tid = threadIdx.x;
    int row = tid;
    if (row < rows) {
        float s = scales[row];
        for (int col = 0; col < cols; col++) {
            mat[row * cols + col] *= s;
        }
    }
}
