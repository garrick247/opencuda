// Probe: ML training patterns — gradient accumulation, AdamW optimizer step,
// dropout with PRNG, softmax backward, cross-attention QKV, and
// mixed-precision training (float32 master weights, float16 compute).

// ------------------------------------------------------------------
// AdamW optimizer step (single element per thread).

__global__ void adamw_step(float *params, float *grads,
                              float *m, float *v,
                              float lr, float beta1, float beta2,
                              float eps, float wd, int step, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float g = grads[gid];
    float mi = m[gid], vi = v[gid];
    // Update moments
    mi = beta1 * mi + (1.0f - beta1) * g;
    vi = beta2 * vi + (1.0f - beta2) * g * g;
    m[gid] = mi; v[gid] = vi;
    // Bias correction
    float m_hat = mi / (1.0f - __powf(beta1, (float)step));
    float v_hat = vi / (1.0f - __powf(beta2, (float)step));
    // Weight update with decoupled weight decay
    float p = params[gid];
    p = p - lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p);
    params[gid] = p;
}

// ------------------------------------------------------------------
// Dropout with LCG PRNG.

__device__ unsigned lcg_next(unsigned state) {
    return state * 1664525u + 1013904223u;
}

__global__ void dropout_fwd(float *out, float *in, float *mask,
                               float prob, unsigned seed, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned state = seed + (unsigned)gid * 2654435761u;
    state = lcg_next(state);
    float r = (float)(state & 0xFFFFFF) / (float)0xFFFFFF;
    float m = (r >= prob) ? 1.0f / (1.0f - prob) : 0.0f;
    mask[gid] = m;
    out[gid] = in[gid] * m;
}

// ------------------------------------------------------------------
// Softmax backward: dL/dx_i = softmax_i * (dL/dy_i - sum_j(softmax_j * dL/dy_j))

__global__ void softmax_backward(float *dx, float *dy, float *softmax_out,
                                    int C, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) return;

    __shared__ float sdot[256];
    // Compute dot(softmax, dy)
    float s = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        int idx = row * C + c;
        s += softmax_out[idx] * dy[idx];
    }
    sdot[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdot[tid] += sdot[tid + stride];
        __syncthreads();
    }
    float dot_val = sdot[0];

    // Compute gradient
    for (int c = tid; c < C; c += blockDim.x) {
        int idx = row * C + c;
        dx[idx] = softmax_out[idx] * (dy[idx] - dot_val);
    }
}

// ------------------------------------------------------------------
// Gradient accumulation (add gradient to accumulator).

__global__ void grad_accum(float *accum, float *grad, float scale, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) accum[gid] += grad[gid] * scale;
}

// ------------------------------------------------------------------
// Gradient clipping by global norm.

__global__ void grad_clip(float *grads, float max_norm, float global_norm, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && global_norm > max_norm) {
        grads[gid] *= max_norm / global_norm;
    }
}

// ------------------------------------------------------------------
// Learning rate warmup + cosine decay schedule.

__device__ float lr_schedule(int step, int warmup_steps, int total_steps,
                               float base_lr, float min_lr) {
    if (step < warmup_steps) {
        return base_lr * (float)step / (float)warmup_steps;
    }
    float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    // Cosine decay: approximate cos with polynomial
    float cos_val = 1.0f - progress;  // simplified
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cos_val);
}

__global__ void apply_lr_schedule(float *lr_out, int step, int warmup,
                                     int total, float base_lr, float min_lr) {
    lr_out[0] = lr_schedule(step, warmup, total, base_lr, min_lr);
}
