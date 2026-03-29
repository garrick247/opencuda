// Probe: large real-world kernel — Sparse Matrix-Vector Multiplication (SpMV)
// in CSR format. Tests complex indexing, inner loops, conditional branches.

__global__ void spmv_csr(float *y, const float *x, const float *values,
                           const int *row_ptr, const int *col_idx,
                           int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int i = start; i < end; i++) {
            sum += values[i] * x[col_idx[i]];
        }
        y[row] = sum;
    }
}

// Segmented reduction for SpMV (one segment per row)
__global__ void spmv_csr_vector(float *y, const float *x,
                                  const float *values, const int *col_idx,
                                  const int *row_ptr, int rows) {
    __shared__ float sdata[256];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 8 + warp_id;  // 8 warps per block

    if (row >= rows) return;

    float sum = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    // Each lane processes multiple elements
    for (int i = start + lane; i < end; i += 32) {
        sum += values[i] * x[col_idx[i]];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}
