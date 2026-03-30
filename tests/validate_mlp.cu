// Neural network building blocks — runtime validated individually.
// Each kernel uses the (float *out, float *a, float *b, int n) signature.

// ReLU
__global__ void nn_relu(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (a[gid] > 0.0f) ? a[gid] : 0.0f;
}

// Leaky ReLU (alpha=0.01)
__global__ void nn_leaky_relu(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (a[gid] > 0.0f) ? a[gid] : 0.01f * a[gid];
}

// SiLU / Swish: x * sigmoid(x)
__global__ void nn_silu(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = a[gid];
        out[gid] = x / (1.0f + expf(-x));
    }
}

// SAXPY: out = alpha * a + b  (alpha = 0.5)
__global__ void nn_saxpy(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = 0.5f * a[gid] + b[gid];
}

// Residual add
__global__ void nn_residual(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] + b[gid];
}

// Element-wise multiply (for gating)
__global__ void nn_gate(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] * b[gid];
}

// Squared difference (for MSE loss)
__global__ void nn_sq_diff(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float d = a[gid] - b[gid];
        out[gid] = d * d;
    }
}

// Clamp to [-1, 1] (for tanh approximation output)
__global__ void nn_clamp_sym(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = a[gid];
        out[gid] = (v < -1.0f) ? -1.0f : (v > 1.0f) ? 1.0f : v;
    }
}
