// Probe: initializer patterns, enum-like defines, multi-dimensional
// index calculations, and miscellaneous patterns not yet covered.

// ------------------------------------------------------------------
// Struct with initializer syntax (C-style init).

struct Point {
    float x, y;
};

__device__ float point_dist_sq(Point a, Point b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

__global__ void nearest_to_origin(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float best = 1e30f;
        for (int i = 0; i < n; i++) {
            float dsq = xs[i] * xs[i] + ys[i] * ys[i];
            if (dsq < best) best = dsq;
        }
        out[0] = best;
    }
}

// ------------------------------------------------------------------
// Enum-like defines used as array size and index.

#define RED   0
#define GREEN 1
#define BLUE  2
#define ALPHA 3
#define NUM_CHANNELS 4

__global__ void channel_sum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int ch = tid % NUM_CHANNELS;
        float v = data[tid];
        // Channel-specific scaling
        float scale = (ch == RED   ? 0.299f :
                       ch == GREEN ? 0.587f :
                       ch == BLUE  ? 0.114f : 0.0f);
        out[tid] = v * scale;
    }
}

// ------------------------------------------------------------------
// Multi-dimensional array indexing: 3D tensor flattened.

__global__ void tensor3d_access(float *out, float *data,
                                  int C, int H, int W) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // tid = (c, h, w) packed
    int w = tid % W;
    int h = (tid / W) % H;
    int c = tid / (W * H);
    if (c < C && h < H && w < W) {
        int idx = c * H * W + h * W + w;
        out[idx + bid * C * H * W] = data[idx];
    }
}

// ------------------------------------------------------------------
// Tiled matrix multiply (small, no shared mem — tests index math).

__global__ void naive_matmul(float *C, float *A, float *B,
                              int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ------------------------------------------------------------------
// Stencil with boundary check.
// Tests that multiple boundary conditions compile correctly.

__global__ void stencil1d(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float left   = (tid > 0)     ? data[tid - 1] : 0.0f;
        float center =                  data[tid];
        float right  = (tid < n - 1) ? data[tid + 1] : 0.0f;
        out[tid] = 0.25f * left + 0.5f * center + 0.25f * right;
    }
}

// ------------------------------------------------------------------
// Kernel with blockDim.y and threadIdx.y.
// Tests that 2D thread indexing is handled.

__global__ void block2d_kernel(float *out, float *data, int W) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x * blockDim.y + y * blockDim.x + x;
    out[idx] = data[idx] * (float)(x + y + 1);
}

// ------------------------------------------------------------------
// Interleaved read/write to same array (no overlap).
// Tests that load addresses and store addresses are tracked separately.

__global__ void interleaved_rw(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        // Swap adjacent pairs: data[2i] ↔ data[2i+1]
        int a = data[tid * 2];
        int b = data[tid * 2 + 1];
        data[tid * 2]     = b;
        data[tid * 2 + 1] = a;
    }
}

// ------------------------------------------------------------------
// Multiple blockIdx dimensions in index calculation.

__global__ void grid3d_index(int *out, int Nx, int Ny) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int idx = z * Ny * Nx + y * Nx + x;
    out[idx] = x + y * 1000 + z * 1000000;
}
