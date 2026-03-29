// Probe: unusual but valid CUDA patterns — kernel calling another kernel
// (dynamic parallelism, parse only), conditional kernel parameters,
// gridDim and blockDim in complex expressions

__global__ void subkernel(float *out, float *in, int offset, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

__global__ void use_all_dims(float *out, float *in) {
    int gx = gridDim.x;
    int gy = gridDim.y;
    int bx = blockDim.x;
    int by = blockDim.y;
    int tid_x = blockIdx.x * bx + threadIdx.x;
    int tid_y = blockIdx.y * by + threadIdx.y;
    int n = gx * bx;
    int m = gy * by;
    if (tid_x < n && tid_y < m) {
        out[tid_y * n + tid_x] = in[tid_y * n + tid_x] * (float)(gx + gy);
    }
}

// 3D grid indexing
__global__ void grid3d(float *out, float *in, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        int idx = z * nx * ny + y * nx + x;
        out[idx] = in[idx] + (float)(x + y + z);
    }
}
