// Probe: size_t, __ldg with multiple types, 2D/3D thread/block indexing,
// and patterns using blockIdx.y / blockIdx.z.

// ------------------------------------------------------------------
// 2D thread grid: row-major matrix element access.

__global__ void mat2d_access(float *out, float *in, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = in[idx] * 2.0f;
    }
}

// ------------------------------------------------------------------
// 3D thread grid: volume element access.

__global__ void vol3d_access(float *out, float *in, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        int idx = z * (ny * nx) + y * nx + x;
        out[idx] = in[idx] + 1.0f;
    }
}

// ------------------------------------------------------------------
// __ldg with int.

__global__ void ldg_int(int *out, const int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]) * 3;
    }
}

// ------------------------------------------------------------------
// __ldg with float.

__global__ void ldg_float(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]) + 1.0f;
    }
}

// ------------------------------------------------------------------
// size_t for index arithmetic.

__global__ void size_t_idx(float *out, float *in, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * in[tid];
    }
}

// ------------------------------------------------------------------
// blockIdx-based work partitioning (each block handles a row).

__global__ void row_sum(float *out, float *in, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[256];
    smem[tid] = (tid < cols) ? in[row * cols + tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[row] = smem[0];
}

// ------------------------------------------------------------------
// blockDim used in formula.

__global__ void block_stride_sum(int *out, int *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int acc = 0;
    for (int i = tid; i < n; i += stride) {
        acc += in[i];
    }
    atomicAdd(out, acc);
}

// ------------------------------------------------------------------
// gridDim: compute global thread count.

__global__ void count_threads(int *out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = blockDim.x * gridDim.x;
    if (tid == 0) out[0] = total;
}
