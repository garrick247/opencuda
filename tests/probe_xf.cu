// Probe: 3D local arrays, struct with 2D array field, complex indexing
// patterns that combine 2D arrays with arithmetic, and more global 2D patterns.

// ------------------------------------------------------------------
// Diagonal sum of a 2D local array.

__global__ void diag_sum_2d(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mat[4][4];
        int base = tid * 16;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                mat[i][j] = in[base + i * 4 + j];
        // Sum of main diagonal
        int s = 0;
        for (int i = 0; i < 4; i++) s += mat[i][i];
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// 2D local array, dynamic row/col.

__global__ void dyn_2d_local(int *out, int *rows_in, int *cols_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mat[4][4];
        // Initialize with tid-dependent values
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                mat[i][j] = (i + tid) * 4 + j;
        int r = rows_in[tid] & 3;
        int c = cols_in[tid] & 3;
        out[tid] = mat[r][c];
    }
}

// ------------------------------------------------------------------
// Global 2D array: float precision table.

#define PREC_ROWS 4
#define PREC_COLS 4
__device__ float g_prec_table[PREC_ROWS][PREC_COLS];

__global__ void init_prec_table(int n) {
    int tid = threadIdx.x;
    if (tid < PREC_ROWS) {
        for (int c = 0; c < PREC_COLS; c++) {
            g_prec_table[tid][c] = (float)(tid) * 0.25f + (float)(c) * 0.0625f;
        }
    }
}

__global__ void use_prec_table(float *out, int *r_idx, int *c_idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = r_idx[tid] & (PREC_ROWS - 1);
        int c = c_idx[tid] & (PREC_COLS - 1);
        out[tid] = g_prec_table[r][c];
    }
}

// ------------------------------------------------------------------
// Compute sum of a 2D global array across all elements.

__global__ void sum_2d_global(float *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < PREC_ROWS; i++)
            for (int j = 0; j < PREC_COLS; j++)
                s += g_prec_table[i][j];
        out[0] = s;
    }
}

// ------------------------------------------------------------------
// Shared 2D array with stride-1 access (no bank conflicts).

__global__ void shared_2d_stride1(int *out, int *in, int n) {
    __shared__ int smem[8][8];  // 8x8 tile
    int tx = threadIdx.x & 7;
    int ty = threadIdx.x >> 3;
    int gid = blockIdx.x * 64 + threadIdx.x;

    smem[ty][tx] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Read from column-major order (stride-8 in smem, could cause bank conflicts)
    if (gid < n) {
        out[gid] = smem[tx][ty];
    }
}

// ------------------------------------------------------------------
// 2D array passed as pointer to device function (decays to pointer).

__device__ int sum_row(int *row, int len) {
    int s = 0;
    for (int i = 0; i < len; i++) s += row[i];
    return s;
}

__global__ void row_sum_device_fn(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < 4) {
        int mat[4][4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                mat[i][j] = tid * 16 + i * 4 + j;
        // Pass mat[tid&3] (a row) to device fn — decays to int*
        out[tid] = sum_row(mat[tid & 3], 4);
    }
}

// ------------------------------------------------------------------
// Matrix multiply using 2D local arrays.

__global__ void matmul_2d_local(float *C, float *A_flat, float *B_flat, int n) {
    int tid = threadIdx.x;
    if (tid < n && n <= 4) {
        float A[4][4], B[4][4], Cout[4][4];
        // Load
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                A[i][j] = A_flat[i * 4 + j];
                B[i][j] = B_flat[i * 4 + j];
            }
        // Multiply
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float s = 0.0f;
                for (int k = 0; k < 4; k++)
                    s += A[i][k] * B[k][j];
                Cout[i][j] = s;
            }
        // Store tid-th row
        for (int j = 0; j < 4; j++)
            C[tid * 4 + j] = Cout[tid][j];
    }
}
