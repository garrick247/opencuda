// Probe: 2D local arrays, 2D shared memory with explicit [row][col] indexing,
// 2D global arrays (multi-kernel), and matrix operations via 2D arrays.

// ------------------------------------------------------------------
// Local 2D array: int tile[4][4] in kernel.

__global__ void local_2d_array(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        int tile[4][4];
        // Fill
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                tile[i][j] = i * 4 + j;
            }
        }
        // Read transposed
        int r = tid / 4;
        int c = tid % 4;
        out[tid] = tile[c][r];  // transposed element
    }
}

// ------------------------------------------------------------------
// 2D global float array for accumulation.

#define N_BINS 8
#define N_VALS 4
__device__ float g_hist[N_BINS][N_VALS];

__global__ void fill_hist(int n) {
    int tid = threadIdx.x;
    if (tid < N_BINS * N_VALS) {
        int bin = tid / N_VALS;
        int val = tid % N_VALS;
        g_hist[bin][val] = (float)(bin * N_VALS + val);
    }
}

__global__ void read_hist(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < N_BINS * N_VALS) {
        int bin = tid / N_VALS;
        int val = tid % N_VALS;
        out[tid] = g_hist[bin][val];
    }
}

// ------------------------------------------------------------------
// 2D shared memory via local array declaration in kernel.

__global__ void smem_2d_tile(float *out, float *in, int width, int height) {
    __shared__ float tile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * 16 + tx;
    int gy = blockIdx.y * 16 + ty;

    // Load
    tile[ty][tx] = (gx < width && gy < height) ? in[gy * width + gx] : 0.0f;
    __syncthreads();

    // Each thread reads the transposed position
    int rx = blockIdx.y * 16 + tx;
    int ry = blockIdx.x * 16 + ty;
    if (rx < height && ry < width) {
        out[ry * height + rx] = tile[tx][ty];
    }
}

// ------------------------------------------------------------------
// Row sum using 2D local array.

__global__ void row_sum_2d(int *out, int *in, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        int sum = 0;
        for (int c = 0; c < cols; c++) {
            sum += in[tid * cols + c];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// 2D global int array: write row by row, read column by column.

#define DIM 4
__device__ int g_square[DIM][DIM];

__global__ void fill_square(int n) {
    int tid = threadIdx.x;
    if (tid < DIM) {
        for (int c = 0; c < DIM; c++) {
            g_square[tid][c] = tid * DIM + c;
        }
    }
}

__global__ void read_square_col(int *out, int col, int n) {
    int tid = threadIdx.x;
    if (tid < DIM && tid < n) {
        out[tid] = g_square[tid][col];
    }
}

// ------------------------------------------------------------------
// Local 2D array for dynamic programming (simple path: store/load).

__global__ void dp_2d(int *out, int *cost, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int dp[4][4];
        // Fill cost as 4x4 from input
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                dp[i][j] = cost[i * 4 + j];

        // Simple row-by-row propagation
        for (int i = 1; i < 4; i++)
            for (int j = 0; j < 4; j++)
                dp[i][j] += dp[i-1][j];

        // Read out last row
        for (int j = 0; j < 4; j++)
            out[j] = dp[3][j];
    }
}
