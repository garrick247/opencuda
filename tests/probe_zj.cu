// Probe: GPU computing classics — parallel scan (Hillis-Steele),
// stream compaction, k-th element selection, matrix multiply with
// shared memory tiling, sparse matrix-vector (CSR), parallel merge,
// and cooperative parallel for pattern.

// ------------------------------------------------------------------
// Hillis-Steele inclusive scan in shared memory.

__global__ void hillis_steele_scan(int *out, int *in, int n) {
    __shared__ int buf[512];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    buf[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();
    for (int d = 1; d < blockDim.x; d <<= 1) {
        int val = (tid >= d) ? buf[tid - d] : 0;
        __syncthreads();
        buf[tid] += val;
        __syncthreads();
    }
    if (gid < n) out[gid] = buf[tid];
}

// ------------------------------------------------------------------
// Tiled matrix multiply (16x16 tiles in shared memory).

#define TILE 16

__global__ void matmul_tiled(float *C, float *A, float *B, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ------------------------------------------------------------------
// CSR sparse matrix-vector multiply.

__global__ void spmv_csr(float *y, float *vals, int *col_idx, int *row_ptr,
                            float *x, int nrows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        float s = 0.0f;
        int start = row_ptr[row];
        int end   = row_ptr[row + 1];
        for (int j = start; j < end; j++) {
            s += vals[j] * x[col_idx[j]];
        }
        y[row] = s;
    }
}

// ------------------------------------------------------------------
// Stream compaction: copy elements > threshold to output using atomicAdd.

__global__ void compact(int *out, int *count, int *in, int threshold, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && in[gid] > threshold) {
        int pos = atomicAdd(count, 1);
        out[pos] = in[gid];
    }
}

// ------------------------------------------------------------------
// Parallel merge of two sorted arrays into one.

__global__ void parallel_merge(int *out, int *a, int na, int *b, int nb) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = na + nb;
    if (gid >= total) return;

    // Binary search to find merge position
    int lo_a = (gid < nb) ? 0 : gid - nb;
    int hi_a = (gid < na) ? gid : na;
    while (lo_a < hi_a) {
        int mid = (lo_a + hi_a) / 2;
        int j = gid - mid;
        if (j > 0 && j <= nb && a[mid] > b[j - 1]) {
            lo_a = mid + 1;
        } else {
            hi_a = mid;
        }
    }
    int i = lo_a;
    int j = gid - i;
    // Pick from a or b
    if (j < 0 || j >= nb || (i < na && a[i] <= b[j])) {
        out[gid] = a[i];
    } else {
        out[gid] = b[j];
    }
}

// ------------------------------------------------------------------
// Image processing: box blur 3x3.

__global__ void box_blur3(float *out, float *in, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                sum += in[ny * W + nx];
                count++;
            }
        }
    }
    out[y * W + x] = sum / (float)count;
}
