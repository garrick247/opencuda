// Shared memory and warp-level kernels for runtime validation.

// Reverse array using shared memory
__global__ void block_reverse(float *out, float *a, float *b, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? a[gid] : 0.0f;
    __syncthreads();
    int rev = blockDim.x - 1 - tid;
    int out_gid = blockIdx.x * blockDim.x + rev;
    if (out_gid < n) out[out_gid] = smem[tid];
}

// Warp reduce sum (each warp outputs one sum)
__global__ void warp_reduce(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float v = (gid < n) ? a[gid] : 0.0f;
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if (lane == 0) out[gid / 32] = v;
}

// Inclusive prefix sum (Hillis-Steele in shared memory)
__global__ void prefix_sum(float *out, float *a, float *b, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? a[gid] : 0.0f;
    __syncthreads();
    for (int d = 1; d < blockDim.x; d <<= 1) {
        float val = (tid >= d) ? smem[tid - d] : 0.0f;
        __syncthreads();
        smem[tid] += val;
        __syncthreads();
    }
    if (gid < n) out[gid] = smem[tid];
}

// Shared memory stencil: out[i] = 0.25*a[i-1] + 0.5*a[i] + 0.25*a[i+1]
__global__ void stencil_1d(float *out, float *a, float *b, int n) {
    __shared__ float smem[258];  // 256 + 2 halo
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    // Load with halo
    smem[tid + 1] = (gid < n) ? a[gid] : 0.0f;
    if (tid == 0) smem[0] = (gid > 0) ? a[gid - 1] : 0.0f;
    if (tid == blockDim.x - 1 || gid == n - 1) smem[tid + 2] = (gid + 1 < n) ? a[gid + 1] : 0.0f;
    __syncthreads();
    if (gid < n)
        out[gid] = 0.25f * smem[tid] + 0.5f * smem[tid + 1] + 0.25f * smem[tid + 2];
}

// Element-wise ReLU
__global__ void relu(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (a[gid] > 0.0f) ? a[gid] : 0.0f;
}

// Sigmoid: out = 1 / (1 + exp(-x))
__global__ void sigmoid(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = 1.0f / (1.0f + expf(-a[gid]));
}
