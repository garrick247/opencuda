// End-to-end MLP: 2-layer network. Weights passed as kernel params.
// input(16) → linear(16,32) → relu → linear(32,4) → argmax

__global__ void linear1(float *out, float *in, float *W, float *bias,
                           int batch, int in_dim, int out_dim) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= batch || col >= out_dim) return;
    float s = 0.0f;
    for (int k = 0; k < in_dim; k++)
        s += in[row * in_dim + k] * W[col * in_dim + k];
    out[row * out_dim + col] = s + bias[col];
}

__global__ void relu_inplace(float *data, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && data[gid] < 0.0f) data[gid] = 0.0f;
}

__global__ void linear2(float *out, float *in, float *W, float *bias,
                           int batch, int in_dim, int out_dim) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= batch || col >= out_dim) return;
    float s = 0.0f;
    for (int k = 0; k < in_dim; k++)
        s += in[row * in_dim + k] * W[col * in_dim + k];
    out[row * out_dim + col] = s + bias[col];
}

__global__ void argmax_row(int *out, float *logits, int cols, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float best = logits[row * cols];
    int best_idx = 0;
    for (int c = 1; c < cols; c++) {
        float v = logits[row * cols + c];
        if (v > best) { best = v; best_idx = c; }
    }
    out[row] = best_idx;
}
