// Probe: complete mini neural network — linear layer, bias add, ReLU,
// linear layer, softmax — all as separate kernels like a real inference
// pipeline. Plus: rotary position embedding (RoPE) and flash-attention
// causal mask generation.

// ------------------------------------------------------------------
// Linear layer: Y = X * W^T (one row per thread).

__global__ void linear_fwd(float *Y, float *X, float *W,
                              int batch, int in_dim, int out_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch && col < out_dim) {
        float s = 0.0f;
        for (int k = 0; k < in_dim; k++) {
            s += X[row * in_dim + k] * W[col * in_dim + k];
        }
        Y[row * out_dim + col] = s;
    }
}

// ------------------------------------------------------------------
// Bias add: Y[i] += bias[i % dim].

__global__ void bias_add(float *Y, float *bias, int dim, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) Y[gid] += bias[gid % dim];
}

// ------------------------------------------------------------------
// ReLU activation (in-place).

__global__ void relu_inplace(float *data, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && data[gid] < 0.0f) data[gid] = 0.0f;
}

// ------------------------------------------------------------------
// Softmax (per-row, shared memory).

__global__ void softmax_fwd(float *out, float *in, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[256];

    // Find max
    float mx = -1e30f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = in[row * cols + c];
        if (v > mx) mx = v;
    }
    smem[tid] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    mx = smem[0];
    __syncthreads();

    // Sum of exp
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        sum += expf(in[row * cols + c] - mx);
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    sum = smem[0];

    // Normalize
    for (int c = tid; c < cols; c += blockDim.x) {
        out[row * cols + c] = expf(in[row * cols + c] - mx) / sum;
    }
}

// ------------------------------------------------------------------
// Rotary Position Embedding (RoPE) — used in LLaMA/GPT-NeoX.
// Applies rotation to pairs of dimensions.

__global__ void rope_fwd(float *out, float *in, int seq_len, int head_dim) {
    int pos = blockIdx.x;   // sequence position
    int d   = threadIdx.x;  // dimension within head (must be < head_dim/2)
    if (pos >= seq_len || d >= head_dim / 2) return;

    // Compute rotation angle
    float freq = 1.0f / __powf(10000.0f, (float)(2 * d) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = __cosf(angle);
    float sin_a = __sinf(angle);

    int idx0 = pos * head_dim + 2 * d;
    int idx1 = pos * head_dim + 2 * d + 1;
    float x0 = in[idx0];
    float x1 = in[idx1];

    out[idx0] = x0 * cos_a - x1 * sin_a;
    out[idx1] = x0 * sin_a + x1 * cos_a;
}

// ------------------------------------------------------------------
// Argmax per row (classification output).

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

// ------------------------------------------------------------------
// Token embedding + positional embedding lookup.

__global__ void embed_tokens(float *out, float *tok_embed, float *pos_embed,
                                int *tokens, int seq_len, int embed_dim) {
    int pos = blockIdx.x;
    int d   = threadIdx.x;
    if (pos >= seq_len || d >= embed_dim) return;
    int tok = tokens[pos];
    out[pos * embed_dim + d] = tok_embed[tok * embed_dim + d]
                              + pos_embed[pos * embed_dim + d];
}

// ------------------------------------------------------------------
// Residual + LayerNorm fused.

__global__ void residual_layernorm(float *out, float *in, float *residual,
                                      float *gamma, float *beta,
                                      float eps, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[256];

    // Residual add
    float val = 0.0f;
    if (tid < D) val = in[row * D + tid] + residual[row * D + tid];

    // Mean
    smem[tid] = (tid < D) ? val : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)D;
    __syncthreads();

    // Variance
    float diff = val - mean;
    smem[tid] = (tid < D) ? diff * diff : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / (float)D;
    float inv_std = 1.0f / sqrtf(var + eps);

    if (tid < D)
        out[row * D + tid] = gamma[tid] * (val - mean) * inv_std + beta[tid];
}
