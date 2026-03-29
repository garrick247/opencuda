// Probe: tricky pointer arithmetic and multi-level indirection
// - T** double pointer
// - pointer comparison
// - NULL comparison: if (ptr == 0)
// - pointer subtraction (ptrdiff)
// - void* cast

__global__ void ptr_diff(int *out, int *start, int *end, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Pointer subtraction as array length
        int len = (int)(end - start);
        out[tid] = len > 0 ? start[tid % len] : 0;
    }
}

__global__ void multi_array(int **matrix, int *out, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        int sum = 0;
        int *row = matrix[tid];
        for (int j = 0; j < cols; j++) {
            sum += row[j];
        }
        out[tid] = sum;
    }
}
