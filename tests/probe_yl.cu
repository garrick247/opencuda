// Probe: register pressure stress (many live vars in one basic block),
// loop with large trip count for unrolling threshold, value used after
// long chain of dependent ops, half2 not supported (test parse tolerance),
// extern C linkage syntax, __launch_bounds__ with two params,
// 3D indexing (blockIdx.z, threadIdx.z, gridDim.z), surface/texture
// surrogate (read-only global pointer pattern), and clock64 usage.

// ------------------------------------------------------------------
// 3D grid/block indexing.

__global__ void kernel3d(float *out, float *in, int Nx, int Ny, int Nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < Nx && y < Ny && z < Nz) {
        int idx = z * Ny * Nx + y * Nx + x;
        out[idx] = in[idx] * 2.0f;
    }
}

// ------------------------------------------------------------------
// __launch_bounds__(maxThreads, minBlocks).

__global__ __launch_bounds__(256, 4)
void launch_bounds_2(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] + 1.0f;
}

// ------------------------------------------------------------------
// Register pressure: 20 live floats in one block.

__global__ void reg_pressure(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // 20 dependent computations — all live until the final sum
        float a0  = v + 0.1f;
        float a1  = v + 0.2f;
        float a2  = v + 0.3f;
        float a3  = v + 0.4f;
        float a4  = v + 0.5f;
        float a5  = v * 1.1f;
        float a6  = v * 1.2f;
        float a7  = v * 1.3f;
        float a8  = v * 1.4f;
        float a9  = v * 1.5f;
        float b0  = a0 * a5;
        float b1  = a1 * a6;
        float b2  = a2 * a7;
        float b3  = a3 * a8;
        float b4  = a4 * a9;
        float c0  = b0 + b1 + b2;
        float c1  = b3 + b4;
        float c2  = c0 + c1;
        float d   = c2 * v;
        float e   = d - a0 - a9;
        out[tid] = e;
    }
}

// ------------------------------------------------------------------
// Loop just above the unroll threshold (trip count = 17, threshold = 16).

__global__ void above_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < 17; i++) s += in[(tid + i) % n];
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Loop just at the unroll threshold (trip count = 16).

__global__ void at_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < 16; i++) s += in[(tid + i) % n];
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// clock / clock64 usage.

__global__ void clock_measure(long long *out, int n) {
    int tid = threadIdx.x;
    long long t0 = clock64();
    // Do some work
    int dummy = tid;
    for (int i = 0; i < 32; i++) dummy = dummy * 3 + 1;
    long long t1 = clock64();
    if (tid < n) out[tid] = t1 - t0 + dummy;
}

// ------------------------------------------------------------------
// Value used after long chain of dependent ops.

__device__ float long_chain(float x) {
    float a = x * x;
    float b = a + x;
    float c = b * a;
    float d = c - b;
    float e = d * x + a;
    float f = e * e - c;
    float g = f + d + b;
    return g;
}

__global__ void long_chain_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = long_chain(in[tid]);
}

// ------------------------------------------------------------------
// gridDim queries.

__global__ void griddim_test(int *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int gx = gridDim.x;
        int gy = gridDim.y;
        int gz = gridDim.z;
        out[tid] = gx * gy * gz;
    }
}

// ------------------------------------------------------------------
// blockDim.y and blockDim.z queries.

__global__ void blockdim_yz(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int bdy = blockDim.y;
        int bdz = blockDim.z;
        out[tid] = bdy * bdz;
    }
}
