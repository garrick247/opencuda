// Probe: real-world kernel patterns — softmax (exp + reduce + normalize),
// batch normalization forward, ReLU/Leaky ReLU/GELU approximation,
// sum reduction with shared memory, matrix transpose with shared padding,
// convolution 1D with constant weights, histogram with shared atomics,
// and elementwise binary ops dispatcher.

// ------------------------------------------------------------------
// Softmax: exp, reduce sum, normalize.

__global__ void softmax(float *out, float *in, int cols, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row * cols + tid >= n || tid >= cols) return;
    float *row_in  = in  + row * cols;
    float *row_out = out + row * cols;
    // 1. Compute max for numerical stability
    __shared__ float smax[256];
    smax[tid] = (tid < cols) ? row_in[tid] : -1e30f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        __syncthreads();
    }
    float mx = smax[0];
    // 2. Exp
    __shared__ float sdata[256];
    float ev = (tid < cols) ? expf(row_in[tid] - mx) : 0.0f;
    sdata[tid] = ev;
    __syncthreads();
    // 3. Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    // 4. Normalize
    if (tid < cols) row_out[tid] = ev / sum;
}

// ------------------------------------------------------------------
// ReLU.

__global__ void relu(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (in[gid] > 0.0f) ? in[gid] : 0.0f;
}

// ------------------------------------------------------------------
// Leaky ReLU (alpha = 0.01).

__global__ void leaky_relu(float *out, float *in, float alpha, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = in[gid];
        out[gid] = (v > 0.0f) ? v : alpha * v;
    }
}

// ------------------------------------------------------------------
// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// Since tanh is not natively available, use the fast approximation via exp.

__device__ float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

__global__ void gelu_approx(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = in[gid];
        float c = 0.7978845608f;  // sqrt(2/pi)
        float inner = c * (x + 0.044715f * x * x * x);
        out[gid] = 0.5f * x * (1.0f + fast_tanh(inner));
    }
}

// ------------------------------------------------------------------
// Matrix transpose with shared memory padding to avoid bank conflicts.

__global__ void transpose(float *out, float *in, int W, int H) {
    __shared__ float tile[16][17];  // 17 to avoid bank conflict
    int bx = blockIdx.x * 16;
    int by = blockIdx.y * 16;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Read tile
    int ix = bx + tx;
    int iy = by + ty;
    if (ix < W && iy < H) tile[ty][tx] = in[iy * W + ix];
    __syncthreads();
    // Write transposed tile
    int ox = by + tx;
    int oy = bx + ty;
    if (ox < H && oy < W) out[oy * H + ox] = tile[tx][ty];
}

// ------------------------------------------------------------------
// 1D convolution with constant weights.

__constant__ float c_filter[9] = {
    0.0625f, 0.125f, 0.1875f, 0.25f, 0.1875f, 0.125f, 0.0625f, 0.0f, 0.0f
};

__global__ void conv1d(float *out, float *in, int filter_len, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float s = 0.0f;
        int half_f = filter_len / 2;
        for (int k = 0; k < filter_len; k++) {
            int idx = gid - half_f + k;
            if (idx >= 0 && idx < n) s += in[idx] * c_filter[k];
        }
        out[gid] = s;
    }
}

// ------------------------------------------------------------------
// Batch normalization forward (simplified — single element per thread).

__global__ void batchnorm_fwd(float *out, float *in, float *mean, float *var,
                                  float *gamma, float *beta, float eps, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int ch = gid % 64;  // assume 64 channels
        float x_hat = (in[gid] - mean[ch]) / sqrtf(var[ch] + eps);
        out[gid] = gamma[ch] * x_hat + beta[ch];
    }
}
