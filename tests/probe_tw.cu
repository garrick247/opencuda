// Probe: AI-style compute patterns — tiled GEMM building blocks,
// batch processing, activation functions, and softmax components.

#define TILE_SIZE 16
#define MAX_CLASSES 10

// ------------------------------------------------------------------
// Tiled matrix multiply building block.

__global__ void tiled_gemm(float *C, const float *A, const float *B,
                             int M, int N, int K) {
    __shared__ float tA[TILE_SIZE][TILE_SIZE];
    __shared__ float tB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------------
// ReLU activation.

__global__ void relu(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = (v > 0.0f) ? v : 0.0f;
    }
}

// ------------------------------------------------------------------
// Leaky ReLU.

__global__ void leaky_relu(float *out, const float *in, float alpha, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = (v > 0.0f) ? v : alpha * v;
    }
}

// ------------------------------------------------------------------
// Softmax numerics: exp then normalize.

__global__ void softmax_exp(float *out, const float *in, int n, int classes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Find max in this sample's row
        float row_max = in[tid * classes + 0];
        for (int c = 1; c < classes; c++) {
            float v = in[tid * classes + c];
            if (v > row_max) row_max = v;
        }
        // Compute exp(x - max) for numerical stability
        float sum = 0.0f;
        for (int c = 0; c < classes; c++) {
            float e = expf(in[tid * classes + c] - row_max);
            out[tid * classes + c] = e;
            sum += e;
        }
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < classes; c++) {
            out[tid * classes + c] *= inv_sum;
        }
    }
}

// ------------------------------------------------------------------
// Element-wise add with broadcast (bias add).

__global__ void bias_add(float *out, const float *in, const float *bias,
                          int batch, int features) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch * features) {
        int feat = tid % features;
        out[tid] = in[tid] + bias[feat];
    }
}

// ------------------------------------------------------------------
// Layer normalization (simplified — no gamma/beta).

__global__ void layer_norm(float *out, const float *in, int n, int d) {
    int tid = blockIdx.x;  // one block per sample
    if (tid < n) {
        int lane = threadIdx.x;
        // Compute mean
        float sum = 0.0f;
        for (int i = lane; i < d; i += blockDim.x) {
            sum += in[tid * d + i];
        }
        // Reduce across threads (simple sequential for small d)
        __shared__ float smem[256];
        smem[lane] = (lane < d) ? sum : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (lane < s) smem[lane] += smem[lane + s];
            __syncthreads();
        }
        float mean = smem[0] / (float)d;
        __syncthreads();
        // Compute variance
        float var = 0.0f;
        for (int i = lane; i < d; i += blockDim.x) {
            float diff = in[tid * d + i] - mean;
            var += diff * diff;
        }
        smem[lane] = (lane < d) ? var : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (lane < s) smem[lane] += smem[lane + s];
            __syncthreads();
        }
        float std = __sqrtf(smem[0] / (float)d + 1e-5f);
        __syncthreads();
        // Normalize
        if (lane < d) {
            out[tid * d + lane] = (in[tid * d + lane] - mean) / std;
        }
    }
}
