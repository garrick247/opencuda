// Probe: Real-world compute patterns from GPGPU benchmarks
// - Matrix transpose with shared memory tiling
// - Sparse matrix-vector multiply (SpMV) CSR format
// - Parallel prefix operations

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_transpose(float *odata, const float *idata,
                                  int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts
    
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = idata[yIndex * width + xIndex];
    }
    __syncthreads();
    
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (xIndex < height && yIndex < width) {
        odata[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void spmv_csr(float *y, const float *values, const int *col_idx,
                          const int *row_ptr, const float *x, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int j = start; j < end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}
