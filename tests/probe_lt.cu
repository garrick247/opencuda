// Probe: matrix multiply (tiled pattern), prefix sum with shared memory,
// histogram with atomic, transpose pattern

// Matrix multiply: C[i][j] = sum_k A[i][k] * B[k][j]
// Naive version (one thread per output element)
__global__ void matmul_naive_small(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Prefix sum (inclusive scan) within a warp
__global__ void warp_prefix_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;
    // Warp-level inclusive scan using shuffle
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (tid >= offset) val += tmp;
    }
    if (tid < n) out[tid] = val;
}

// Histogram: count occurrences of values in [0, NBINS)
#define NBINS 16
__global__ void histogram(int *hist, int *in, int n) {
    __shared__ int local_hist[NBINS];
    int tid = threadIdx.x;
    if (tid < NBINS) local_hist[tid] = 0;
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        int bin = in[gid] % NBINS;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();
    if (tid < NBINS) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

// Matrix transpose: out[j][i] = in[i][j]
__global__ void transpose(float *out, float *in, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}
