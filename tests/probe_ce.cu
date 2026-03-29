// Probe: __device__ function returning a pointer
// - __device__ function returning pointer to global memory
// - __device__ function returning null (0/nullptr)
// - Pointer arithmetic in device function
// - Pointer returned from function used directly for indexing

__device__ float* get_row(float *matrix, int row, int cols) {
    return matrix + row * cols;
}

__global__ void row_access(float *out, float *matrix, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        float *row = get_row(matrix, tid, cols);
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += row[j];
        }
        out[tid] = sum;
    }
}

// Pointer comparison
__global__ void ptr_compare(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // This tests pointer arithmetic: &a[n] should equal a + n
        int *end_a = a + n;
        int *cur = a;
        int count = 0;
        while (cur < end_a) {
            count++;
            cur++;
        }
        out[tid] = count;
    }
}
