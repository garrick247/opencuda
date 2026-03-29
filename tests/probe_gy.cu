// Probe: Real ML workload patterns — GEMM building blocks,
// transpose, element-wise ops with broadcasting (1D and 2D)

// Element-wise fused multiply-add with broadcast bias
__global__ void linear_fwd(float *out, const float *in, const float *weight,
                             const float *bias, int batch, int in_dim, int out_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // batch element
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // output feature
    if (row < batch && col < out_dim) {
        float sum = bias[col];  // broadcast bias
        for (int k = 0; k < in_dim; k++) {
            sum += in[row * in_dim + k] * weight[col * in_dim + k];
        }
        out[row * out_dim + col] = sum;
    }
}

// ReLU activation
__global__ void relu(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = v > 0.0f ? v : 0.0f;
    }
}

// Sigmoid activation
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void sigmoid_kernel(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = sigmoid(in[tid]);
    }
}

// Cross-entropy loss gradient (softmax output vs one-hot target)
__global__ void ce_loss_grad(float *grad, const float *softmax_out,
                               const int *targets, int batch, int classes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch * classes) {
        int b = tid / classes;
        int c = tid % classes;
        float s = softmax_out[tid];
        float t = (c == targets[b]) ? 1.0f : 0.0f;
        grad[tid] = (s - t) / (float)batch;
    }
}
