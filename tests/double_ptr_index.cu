// Regression: T**[i][j] double-subscript lvalue assignment
// Without fix: _parse_lvalue_or_expr computed addr = matrix + tid*8 (address
//   of the row pointer), then returned without handling the second [j] subscript
//   → SEMI expected but got LBRACKET when the compound assignment tried to parse.
// Fix: in the lvalue parser's single-subscript else branch, detect a second '['
//   when the element type is PtrTy, load the row pointer, then index into it.

__global__ void double_ptr_test(float **matrix, int rows, int cols, float scale) {
    int tid = threadIdx.x;
    if (tid < rows) {
        for (int j = 0; j < cols; j++) {
            matrix[tid][j] *= scale;
        }
    }
}

// Read via T**[i][j] (rvalue path)
__global__ void double_ptr_read(float *out, float **rows, int n, int cols) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
            sum += rows[tid][j];
        out[tid] = sum;
    }
}
