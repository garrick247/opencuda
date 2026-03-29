// Probe: gridDim usage, global thread ID patterns, stride patterns,
// atomicAdd return value, multi-block accumulation

__global__ void global_tid_stride(float *out, float *in, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = gtid; i < n; i += stride) {
        sum += in[i];
    }
    out[gtid] = sum;
}

// atomicAdd returns old value — use it as an index
__global__ void compaction(int *out, int *count, int *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n && in[tid] > 0) {
        int slot = atomicAdd(count, 1);
        out[slot] = in[tid];
    }
}

// 2D grid indexing
__global__ void matmul_global(float *C, float *A, float *B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// gridDim.y, blockDim.z, threadIdx.z usage
__global__ void dim3_access(int *out) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int gx = gridDim.x;
    int gy = gridDim.y;
    int gz = gridDim.z;
    out[0] = x + y * gx + z * gx * gy + gz;
}
