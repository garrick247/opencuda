// Probe: 2D/3D thread indexing, __constant__ arrays, global array writes
// from device, gridDim.y/z, and kernel with 2D blocks.

// ------------------------------------------------------------------
// 2D thread indexing: row × col for matrix ops.

__global__ void mat_scale(float *out, float *in, float scale, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[row * cols + col] = in[row * cols + col] * scale;
    }
}

// ------------------------------------------------------------------
// 3D thread indexing: volume traversal.

__global__ void vol_fill(float *out, float val, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        out[(z * ny + y) * nx + x] = val;
    }
}

// ------------------------------------------------------------------
// __constant__ array: read-only data in constant memory.

__constant__ float c_kernel[9];  // 3x3 convolution kernel

__global__ void conv3x3(float *out, float *in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int kid = (ky + 1) * 3 + (kx + 1);
                sum += in[(y + ky) * width + (x + kx)] * c_kernel[kid];
            }
        }
        out[y * width + x] = sum;
    }
}

// ------------------------------------------------------------------
// gridDim.y and gridDim.z access.

__global__ void gridinfo(int *out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        out[0] = gridDim.x;
        out[1] = gridDim.y;
        out[2] = gridDim.z;
        out[3] = blockDim.x;
        out[4] = blockDim.y;
        out[5] = blockDim.z;
    }
}

// ------------------------------------------------------------------
// 2D shared memory: tile for matrix multiply.

__global__ void tiled_matmul(float *C, float *A, float *B, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + 15) / 16; t++) {
        int aCol = t * 16 + threadIdx.x;
        int bRow = t * 16 + threadIdx.y;

        if (row < N && aCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ------------------------------------------------------------------
// __constant__ int array: lookup table.

__constant__ int c_lut[16];

__global__ void lut_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = in[tid] & 15;
        out[tid] = c_lut[idx];
    }
}
