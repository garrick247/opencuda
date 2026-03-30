// Probe: production-style kernels — vector dot product, L2 distance,
// sparse-dense vector multiply (CSR format), stream compaction with scan,
// k-nearest neighbor distance, prefix-sum then scatter, per-warp histogram,
// online Welford variance, and multi-head attention score (QK^T/sqrt(d)).

// ------------------------------------------------------------------
// Vector dot product with block reduction.

__global__ void dot_product(float *out, float *a, float *b, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float prod = (gid < n) ? a[gid] * b[gid] : 0.0f;
    smem[tid] = prod;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// ------------------------------------------------------------------
// L2 distance between two vectors (per-block partial).

__global__ void l2_distance(float *out, float *a, float *b, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float diff = (gid < n) ? (a[gid] - b[gid]) : 0.0f;
    smem[tid] = diff * diff;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// ------------------------------------------------------------------
// Online Welford variance (per-thread over strided elements).

__global__ void welford_var(float *out_mean, float *out_var, float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    for (int i = tid; i < n; i += stride) {
        count++;
        float delta = in[i] - mean;
        mean += delta / (float)count;
        float delta2 = in[i] - mean;
        m2 += delta * delta2;
    }
    out_mean[tid] = mean;
    out_var[tid] = (count > 1) ? m2 / (float)(count - 1) : 0.0f;
}

// ------------------------------------------------------------------
// Multi-head attention score: QK^T / sqrt(d) for one head.
// Q: [seq_q, d], K: [seq_k, d], out: [seq_q, seq_k]

__global__ void attention_score(float *out, float *Q, float *K,
                                   int seq_q, int seq_k, int d) {
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < seq_q && k < seq_k) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) {
            s += Q[q * d + i] * K[k * d + i];
        }
        float inv_sqrt_d = 1.0f / sqrtf((float)d);
        out[q * seq_k + k] = s * inv_sqrt_d;
    }
}

// ------------------------------------------------------------------
// Prefix-sum then scatter: move elements to sorted positions.

__global__ void scatter_by_prefix(int *out, int *in, int *prefix, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int dest = prefix[gid];
        if (dest >= 0 && dest < n) out[dest] = in[gid];
    }
}

// ------------------------------------------------------------------
// Per-warp histogram: each warp computes local histogram of 4 bins.

__global__ void warp_histogram(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int v = (gid < n) ? (in[gid] & 3) : -1;  // 4 bins: 0,1,2,3

    // Count per bin using ballot
    unsigned mask0 = __ballot_sync(0xFFFFFFFF, v == 0);
    unsigned mask1 = __ballot_sync(0xFFFFFFFF, v == 1);
    unsigned mask2 = __ballot_sync(0xFFFFFFFF, v == 2);
    unsigned mask3 = __ballot_sync(0xFFFFFFFF, v == 3);

    if (lane == 0 && gid < n) {
        atomicAdd(&out[warp_id * 4 + 0], __popc(mask0));
        atomicAdd(&out[warp_id * 4 + 1], __popc(mask1));
        atomicAdd(&out[warp_id * 4 + 2], __popc(mask2));
        atomicAdd(&out[warp_id * 4 + 3], __popc(mask3));
    }
}

// ------------------------------------------------------------------
// Elementwise add with alpha scaling: out = alpha*a + b.

__global__ void axpy(float *out, float alpha, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = alpha * a[gid] + b[gid];
}

// ------------------------------------------------------------------
// Running exponential moving average (EMA).

__global__ void ema(float *out, float *in, float alpha, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float ema_val = in[0];
        for (int i = 1; i <= tid; i++) {
            ema_val = alpha * in[i] + (1.0f - alpha) * ema_val;
        }
        out[tid] = ema_val;
    }
}

// ------------------------------------------------------------------
// Sigmoid activation.

__global__ void sigmoid(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = in[gid];
        out[gid] = 1.0f / (1.0f + expf(-v));
    }
}
