// Probe: Real-world CUDA patterns from ML/DL kernels
// - Softmax forward pass
// - ReLU with derivative (backward pass pattern)
// - Layer normalization building block

__global__ void softmax_row(float *out, float *in, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float *row_in = in + row * cols;
    float *row_out = out + row * cols;
    
    // Find max for numerical stability
    float mx = row_in[0];
    for (int j = 1; j < cols; j++) {
        if (row_in[j] > mx) mx = row_in[j];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        row_out[j] = expf(row_in[j] - mx);
        sum += row_out[j];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < cols; j++) {
        row_out[j] *= inv_sum;
    }
}

__global__ void relu_backward(float *grad_in, float *grad_out, float *fwd_in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        grad_in[tid] = (fwd_in[tid] > 0.0f) ? grad_out[tid] : 0.0f;
    }
}
