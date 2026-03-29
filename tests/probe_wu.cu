// Probe: convolution, matrix-vector, loop unrolling edge cases,
// multi-block cooperative writes, and integer edge cases.

// ------------------------------------------------------------------
// 1D convolution: each thread computes one output.

#define FILTER_LEN 5
__constant__ float c_filter[FILTER_LEN];

__global__ void conv1d(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float sum = 0.0f;
        for (int k = 0; k < FILTER_LEN; k++) {
            int src = gid - k + FILTER_LEN / 2;
            if (src >= 0 && src < n) {
                sum += in[src] * c_filter[k];
            }
        }
        out[gid] = sum;
    }
}

// ------------------------------------------------------------------
// Matrix-vector multiply: y = A * x, A is rows×cols.

__global__ void matvec(float *y, float *A, float *x, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float acc = 0.0f;
        for (int j = 0; j < cols; j++) {
            acc += A[row * cols + j] * x[j];
        }
        y[row] = acc;
    }
}

// ------------------------------------------------------------------
// Outer product: C[i][j] = a[i] * b[j].

__global__ void outer_product(float *C, float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        C[i * n + j] = a[i] * b[j];
    }
}

// ------------------------------------------------------------------
// Reduction with multiple warps: block-wide max using shared memory + warp shuffle.

__global__ void block_max(int *out, int *in, int n) {
    __shared__ int warp_max[8];  // up to 8 warps per block (256 threads)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane  = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;

    int v = (gid < n) ? in[gid] : -2147483648;

    // Warp-level max via shfl_xor
    v = max(v, __shfl_xor_sync(0xFFFFFFFF, v, 16));
    v = max(v, __shfl_xor_sync(0xFFFFFFFF, v,  8));
    v = max(v, __shfl_xor_sync(0xFFFFFFFF, v,  4));
    v = max(v, __shfl_xor_sync(0xFFFFFFFF, v,  2));
    v = max(v, __shfl_xor_sync(0xFFFFFFFF, v,  1));

    if (lane == 0) warp_max[warpid] = v;
    __syncthreads();

    // Final reduce among warp leaders (tid 0 only)
    if (threadIdx.x == 0) {
        int m = warp_max[0];
        for (int w = 1; w < 8; w++) {
            m = max(m, warp_max[w]);
        }
        out[blockIdx.x] = m;
    }
}

// ------------------------------------------------------------------
// Unrolled dot product of exactly 8 elements.

__global__ void dot8(float *out, float *a, float *b, int n) {
    int base = blockIdx.x * 8;
    if (base + 7 < n) {
        float sum = 0.0f;
        // Manual unroll 8
        sum += a[base + 0] * b[base + 0];
        sum += a[base + 1] * b[base + 1];
        sum += a[base + 2] * b[base + 2];
        sum += a[base + 3] * b[base + 3];
        sum += a[base + 4] * b[base + 4];
        sum += a[base + 5] * b[base + 5];
        sum += a[base + 6] * b[base + 6];
        sum += a[base + 7] * b[base + 7];
        out[blockIdx.x] = sum;
    }
}

// ------------------------------------------------------------------
// Modular arithmetic: compute a^b mod m using repeated squaring.

__device__ long long powmod(long long base, long long exp, long long mod) {
    long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void powmod_kernel(long long *out, long long *bases, long long *exps, long long mod, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = powmod(bases[tid], exps[tid], mod);
    }
}

// ------------------------------------------------------------------
// Counting sort: count histogram then accumulate (count phase only).

__global__ void count_sort_histogram(int *hist, int *in, int n, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid] % range;
        atomicAdd(&hist[v], 1);
    }
}

// ------------------------------------------------------------------
// RNG: simple xorshift32 per thread.

__device__ unsigned int xorshift32(unsigned int state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__global__ void rng_kernel(unsigned int *out, unsigned int *seeds, int n, int iters) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int s = seeds[tid];
        for (int i = 0; i < iters; i++) {
            s = xorshift32(s);
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Bilinear interpolation: sample a 2D grid at fractional coordinates.

__global__ void bilinear_interp(float *out, float *grid, float *coords_x,
                                float *coords_y, int width, int height, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fx = coords_x[tid];
        float fy = coords_y[tid];

        int x0 = (int)fx;
        int y0 = (int)fy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // Clamp
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 >= width)  x1 = width  - 1;
        if (y1 >= height) y1 = height - 1;

        float wx = fx - (float)x0;
        float wy = fy - (float)y0;

        float v00 = grid[y0 * width + x0];
        float v10 = grid[y0 * width + x1];
        float v01 = grid[y1 * width + x0];
        float v11 = grid[y1 * width + x1];

        out[tid] = (1.0f - wx) * (1.0f - wy) * v00
                 +         wx  * (1.0f - wy) * v10
                 + (1.0f - wx) *         wy  * v01
                 +         wx  *         wy  * v11;
    }
}
