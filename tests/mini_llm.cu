// Mini LLM: complete transformer decoder inference pipeline.
// Tests the compiler's ability to handle a realistic ML workload
// with all the building blocks in one translation unit.

// ------------------------------------------------------------------
// Token embedding + positional encoding.

__global__ void embed(float *out, float *tok_table, float *pos_table,
                         int *tokens, int seq_len, int dim) {
    int pos = blockIdx.x;
    int d   = threadIdx.x;
    if (pos < seq_len && d < dim) {
        out[pos * dim + d] = tok_table[tokens[pos] * dim + d]
                           + pos_table[pos * dim + d];
    }
}

// ------------------------------------------------------------------
// RMSNorm: x_hat = x / rms(x), y = gamma * x_hat.

__global__ void rmsnorm(float *out, float *in, float *weight,
                           float eps, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[256];

    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = in[row * dim + i];
        ss += v * v;
    }
    smem[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / (float)dim + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        out[idx] = weight[i] * in[idx] * inv_rms;
    }
}

// ------------------------------------------------------------------
// QKV projection: three matrix multiplies packed.

__global__ void qkv_proj(float *Q, float *K, float *V,
                            float *X, float *Wq, float *Wk, float *Wv,
                            int seq_len, int dim, int head_dim) {
    int pos = blockIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len || d >= head_dim) return;

    float q = 0.0f, k = 0.0f, v = 0.0f;
    for (int i = 0; i < dim; i++) {
        float xi = X[pos * dim + i];
        q += xi * Wq[d * dim + i];
        k += xi * Wk[d * dim + i];
        v += xi * Wv[d * dim + i];
    }
    Q[pos * head_dim + d] = q;
    K[pos * head_dim + d] = k;
    V[pos * head_dim + d] = v;
}

// ------------------------------------------------------------------
// RoPE (rotary position embedding).

__global__ void rope(float *Q, float *K, int seq_len, int head_dim) {
    int pos = blockIdx.x;
    int d   = threadIdx.x;
    if (pos >= seq_len || d >= head_dim / 2) return;

    float freq = 1.0f / __powf(10000.0f, (float)(2 * d) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = __cosf(angle);
    float sin_a = __sinf(angle);

    int i0 = pos * head_dim + 2 * d;
    int i1 = i0 + 1;

    // Apply to Q
    float q0 = Q[i0], q1 = Q[i1];
    Q[i0] = q0 * cos_a - q1 * sin_a;
    Q[i1] = q0 * sin_a + q1 * cos_a;

    // Apply to K
    float k0 = K[i0], k1 = K[i1];
    K[i0] = k0 * cos_a - k1 * sin_a;
    K[i1] = k0 * sin_a + k1 * cos_a;
}

// ------------------------------------------------------------------
// Attention: score = QK^T/sqrt(d), apply causal mask.

__global__ void attention_scores(float *scores, float *Q, float *K,
                                    int seq_len, int head_dim) {
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= seq_len || k >= seq_len) return;

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++)
        dot += Q[q * head_dim + d] * K[k * head_dim + d];
    dot *= rsqrtf((float)head_dim);

    // Causal mask
    if (k > q) dot = -1e9f;
    scores[q * seq_len + k] = dot;
}

// ------------------------------------------------------------------
// Softmax per row.

__global__ void softmax_row(float *out, float *in, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[256];

    // Max
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

    // Sum exp
    float se = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x)
        se += expf(in[row * cols + c] - mx);
    smem[tid] = se;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum = smem[0];

    // Normalize
    for (int c = tid; c < cols; c += blockDim.x)
        out[row * cols + c] = expf(in[row * cols + c] - mx) / sum;
}

// ------------------------------------------------------------------
// Attention output: out = scores * V.

__global__ void attention_output(float *out, float *scores, float *V,
                                    int seq_len, int head_dim) {
    int pos = blockIdx.y * blockDim.y + threadIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len || d >= head_dim) return;

    float s = 0.0f;
    for (int k = 0; k < seq_len; k++)
        s += scores[pos * seq_len + k] * V[k * head_dim + d];
    out[pos * head_dim + d] = s;
}

// ------------------------------------------------------------------
// Feed-forward: SwiGLU — y = (xW1 * silu(xW_gate)) * W2.

__global__ void ffn_gate(float *gate_out, float *up_out,
                            float *X, float *W_gate, float *W_up,
                            int seq_len, int dim, int ff_dim) {
    int pos = blockIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len || d >= ff_dim) return;

    float g = 0.0f, u = 0.0f;
    for (int i = 0; i < dim; i++) {
        float xi = X[pos * dim + i];
        g += xi * W_gate[d * dim + i];
        u += xi * W_up[d * dim + i];
    }
    // SiLU on gate
    float silu_g = g / (1.0f + expf(-g));
    gate_out[pos * ff_dim + d] = silu_g * u;
}

// ------------------------------------------------------------------
// Down projection: y = xW_down + residual.

__global__ void ffn_down_residual(float *out, float *X, float *W_down,
                                     float *residual,
                                     int seq_len, int dim, int ff_dim) {
    int pos = blockIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len || d >= dim) return;

    float s = 0.0f;
    for (int i = 0; i < ff_dim; i++)
        s += X[pos * ff_dim + i] * W_down[d * ff_dim + i];
    out[pos * dim + d] = s + residual[pos * dim + d];
}

// ------------------------------------------------------------------
// Final: argmax for next token prediction.

__global__ void argmax_token(int *out, float *logits, int vocab_size) {
    int pos = blockIdx.x;
    float best = logits[pos * vocab_size];
    int best_idx = 0;
    for (int v = 1; v < vocab_size; v++) {
        float l = logits[pos * vocab_size + v];
        if (l > best) { best = l; best_idx = v; }
    }
    out[pos] = best_idx;
}
