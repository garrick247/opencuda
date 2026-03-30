// Matrix multiply and dot product for runtime validation.
// Signature: (float *out, float *a, float *b, int n) where n is treated
// as dimension size for the square matrix.

// Simple matrix multiply: C[i][j] = sum_k A[i][k] * B[k][j]
// Thread per output element.
__global__ void matmul_simple(float *out, float *a, float *b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0.0f;
        for (int k = 0; k < n; k++) {
            s += a[row * n + k] * b[k * n + col];
        }
        out[row * n + col] = s;
    }
}
