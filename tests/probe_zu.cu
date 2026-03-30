// Probe: final push — GAN training discriminator step, quantization
// (float32→int8), dequantization (int8→float32), top-k selection,
// batch GEMM (batched matmul), and flash-attention score computation.

// ------------------------------------------------------------------
// Quantization: float32 → int8 with scale and zero point.

__global__ void quantize_f32_i8(signed char *out, float *in,
                                   float scale, int zero_point, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int q = __float2int_rn(in[gid] / scale) + zero_point;
    // Clamp to [-128, 127]
    q = (q < -128) ? -128 : (q > 127) ? 127 : q;
    out[gid] = (signed char)q;
}

// ------------------------------------------------------------------
// Dequantization: int8 → float32.

__global__ void dequantize_i8_f32(float *out, signed char *in,
                                     float scale, int zero_point, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = ((float)((int)in[gid] - zero_point)) * scale;
}

// ------------------------------------------------------------------
// Top-k: find k-th largest via iterative threshold.

__global__ void topk_threshold(int *count_out, float *data,
                                  float threshold, int n) {
    __shared__ int cnt[1];
    int tid = threadIdx.x;
    if (tid == 0) cnt[0] = 0;
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n && data[gid] >= threshold) {
        atomicAdd(&cnt[0], 1);
    }
    __syncthreads();
    if (tid == 0) atomicAdd(count_out, cnt[0]);
}

// ------------------------------------------------------------------
// Batch GEMM: C[b] = A[b] * B[b] for each batch.

__global__ void batch_gemm(float *C, float *A, float *B,
                              int M, int N, int K, int batch) {
    int b   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || row >= M || col >= N) return;
    int stride_a = M * K;
    int stride_b = K * N;
    int stride_c = M * N;
    float s = 0.0f;
    for (int k = 0; k < K; k++) {
        s += A[b * stride_a + row * K + k] * B[b * stride_b + k * N + col];
    }
    C[b * stride_c + row * N + col] = s;
}

// ------------------------------------------------------------------
// Flash-attention-style: QK^T with causal mask and online softmax.

__global__ void flash_attn_score(float *out, float *Q, float *K,
                                    int seq_len, int head_dim) {
    int q_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int k_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_pos >= seq_len || k_pos >= seq_len) return;

    // Compute dot product Q[q] · K[k]
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += Q[q_pos * head_dim + d] * K[k_pos * head_dim + d];
    }
    // Scale by 1/sqrt(d)
    dot *= rsqrtf((float)head_dim);
    // Causal mask
    if (k_pos > q_pos) dot = -1e9f;
    out[q_pos * seq_len + k_pos] = dot;
}

// ------------------------------------------------------------------
// GeLU backward: dL/dx = dL/dy * gelu'(x)
// gelu'(x) ≈ 0.5 * (1 + tanh(a)) + 0.5 * x * (1 - tanh^2(a)) * a'
// where a = sqrt(2/pi) * (x + 0.044715 * x^3), a' = sqrt(2/pi) * (1 + 0.134145 * x^2)

__global__ void gelu_backward(float *dx, float *dy, float *x_in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float x = x_in[gid];
    float c = 0.7978845608f;  // sqrt(2/pi)
    float x3 = x * x * x;
    float inner = c * (x + 0.044715f * x3);
    // tanh approximation
    float t = inner;
    float t2 = t * t;
    float tanh_val = t * (27.0f + t2) / (27.0f + 9.0f * t2);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float inner_deriv = c * (1.0f + 0.134145f * x * x);
    float gelu_grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * inner_deriv;
    dx[gid] = dy[gid] * gelu_grad;
}

// ------------------------------------------------------------------
// Weight initialization: Xavier/Glorot uniform via LCG PRNG.

__global__ void xavier_init(float *W, int fan_in, int fan_out,
                               unsigned seed, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    // LCG PRNG
    unsigned state = seed + (unsigned)gid * 2654435761u;
    state = state * 1664525u + 1013904223u;
    float u = (float)(state & 0xFFFFFF) / (float)0xFFFFFF;  // [0, 1)
    // Xavier range: [-limit, +limit] where limit = sqrt(6 / (fan_in + fan_out))
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    W[gid] = (2.0f * u - 1.0f) * limit;
}
