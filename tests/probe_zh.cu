// Probe: ML inference patterns — layernorm, embedding lookup, causal mask,
// top-k via partial sort, cross-entropy loss, and residual connection.

// ------------------------------------------------------------------
// Layer normalization.

__global__ void layernorm(float *out, float *in, float *gamma, float *beta,
                             float eps, int D, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) return;

    __shared__ float smem[256];

    // Compute mean
    float s = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) s += in[row * D + i];
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float mean = smem[0] / (float)D;
    __syncthreads();

    // Compute variance
    s = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = in[row * D + i] - mean;
        s += diff * diff;
    }
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float var = smem[0] / (float)D;
    float inv_std = 1.0f / sqrtf(var + eps);

    // Normalize
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = row * D + i;
        float x_hat = (in[idx] - mean) * inv_std;
        out[idx] = gamma[i] * x_hat + beta[i];
    }
}

// ------------------------------------------------------------------
// Embedding lookup.

__global__ void embedding_lookup(float *out, float *table, int *indices,
                                    int D, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int idx = indices[gid / D];
        int dim = gid % D;
        out[gid] = table[idx * D + dim];
    }
}

// ------------------------------------------------------------------
// Causal attention mask: out[i][j] = (j <= i) ? 0.0f : -1e9f

__global__ void causal_mask(float *out, int seq_len) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len && j < seq_len) {
        out[i * seq_len + j] = (j <= i) ? 0.0f : -1000000000.0f;
    }
}

// ------------------------------------------------------------------
// Cross-entropy loss (per-sample).

__global__ void cross_entropy(float *loss, float *logits, int *targets,
                                int C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int target = targets[tid];
        float max_logit = logits[tid * C];
        for (int c = 1; c < C; c++) {
            float l = logits[tid * C + c];
            if (l > max_logit) max_logit = l;
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_exp += expf(logits[tid * C + c] - max_logit);
        }
        float log_sum_exp = max_logit + logf(sum_exp);
        loss[tid] = log_sum_exp - logits[tid * C + target];
    }
}

// ------------------------------------------------------------------
// Residual connection: out = in + residual.

__global__ void residual_add(float *out, float *in, float *residual, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] + residual[gid];
}

// ------------------------------------------------------------------
// SiLU (Swish) activation: x * sigmoid(x).

__global__ void silu(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = in[gid];
        out[gid] = x / (1.0f + expf(-x));
    }
}

// ------------------------------------------------------------------
// RMSNorm (used in LLaMA).

__global__ void rmsnorm(float *out, float *in, float *weight,
                           float eps, int D, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) return;

    __shared__ float smem[256];

    // Compute sum of squares
    float s = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = in[row * D + i];
        s += v * v;
    }
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float rms = sqrtf(smem[0] / (float)D + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = row * D + i;
        out[idx] = in[idx] * inv_rms * weight[i];
    }
}
