// Block-wise softmax: each block computes softmax of one row.
// Two-pass (max-reduce, then sum-reduce), tree reduction in shared memory.
//
// Pipeline target: OpenCUDA -> OpenPTXas -> SM_120 cubin.
// NVIDIA nvcc / ptxas are NOT used at any stage.

#define BLOCK 256

__global__ void softmax_rowwise(float *inp, float *out, int n_cols)
{
    __shared__ float smem[BLOCK];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    float x = 0.0f;
    if (tid < n_cols) {
        x = inp[row * n_cols + tid];
    }

    // Pass 1: reduce max.
    float row_max = x;
    if (tid >= n_cols) {
        row_max = -1e30f;  // neutral element for max — outside the active lanes
    }
    smem[tid] = row_max;
    __syncthreads();
    int stride = BLOCK / 2;
    while (stride > 0) {
        if (tid < stride) {
            float other = smem[tid + stride];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
        stride = stride / 2;
    }
    row_max = smem[0];
    __syncthreads();

    // Pass 2: exp(x - max), reduce sum.
    float ex = 0.0f;
    if (tid < n_cols) {
        ex = expf(x - row_max);
    }
    smem[tid] = ex;
    __syncthreads();
    stride = BLOCK / 2;
    while (stride > 0) {
        if (tid < stride) {
            smem[tid] = smem[tid] + smem[tid + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    float row_sum = smem[0];

    // Normalize and write out.
    if (tid < n_cols) {
        out[row * n_cols + tid] = ex / row_sum;
    }
}
