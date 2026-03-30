// Compute-intensive kernels for runtime validation.

// Element-wise multiply
__global__ void vector_mul(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] * b[gid];
}

// Negate
__global__ void vector_neg(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = -a[gid];
}

// Square each element
__global__ void vector_sq(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] * a[gid];
}

// Fused multiply-add: out = a * b + a (not using b for add, using a)
__global__ void vector_fma(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] * b[gid] + a[gid];
}

// Max of two arrays
__global__ void vector_max(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (a[gid] > b[gid]) ? a[gid] : b[gid];
}

// Absolute value
__global__ void vector_abs(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (a[gid] < 0.0f) ? -a[gid] : a[gid];
}

// Clamp to [0, 1]
__global__ void vector_clamp01(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = a[gid];
        out[gid] = (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
    }
}
