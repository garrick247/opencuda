// Probe: complex indexing, 2D arrays, stencil patterns, and
// address computation edge cases.

// ------------------------------------------------------------------
// Row-major 2D array index: [row][col] linearized.

__global__ void rowmaj_access(float *out, float *in,
                               int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        // Read from transposed position (col, row)
        out[row * cols + col] = in[col * rows + row];
    }
}

// ------------------------------------------------------------------
// 5-point stencil (2D Laplacian).

__global__ void stencil5pt(float *out, float *in, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        float center = in[y * nx + x];
        float left   = in[y * nx + (x - 1)];
        float right  = in[y * nx + (x + 1)];
        float up     = in[(y - 1) * nx + x];
        float down   = in[(y + 1) * nx + x];
        out[y * nx + x] = 0.25f * (left + right + up + down) - center;
    }
}

// ------------------------------------------------------------------
// Diagonal access pattern.

__global__ void diag_access(float *out, float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Access along diagonal: in[i*(n+1)]
        int diag_idx = tid * (n + 1);
        if (diag_idx < n * n) {
            out[tid] = in[diag_idx];
        }
    }
}

// ------------------------------------------------------------------
// Spiral-like index computation.

__global__ void spiral_idx(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mid = n / 2;
        // Reindex: first half reads from end, second half reads from start
        int src = (tid < mid) ? (n - 1 - tid) : (tid - mid);
        out[tid] = in[src];
    }
}

// ------------------------------------------------------------------
// Block-tiled copy with shared memory and sync.

__global__ void tiled_copy(float *out, float *in, int n) {
    __shared__ float tile[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) tile[tid] = in[gid];
    else tile[tid] = 0.0f;
    __syncthreads();
    // Reversed within tile
    int rev = blockDim.x - 1 - tid;
    if (gid < n) out[gid] = tile[rev];
}

// ------------------------------------------------------------------
// Gather: read from scattered locations.

__global__ void gather(float *out, float *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int src = idx[tid];
        // Bounds-checked gather
        out[tid] = (src >= 0 && src < n) ? in[src] : 0.0f;
    }
}

// ------------------------------------------------------------------
// Scatter with atomic: multiple threads may write same location.

__global__ void scatter_atomic(int *out, int *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int dst = idx[tid] % n;
        atomicAdd(out + dst, in[tid]);
    }
}

// ------------------------------------------------------------------
// Prefix sum (scan) within a warp.

__global__ void warp_scan(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    unsigned mask = 0xFFFFFFFFu;
    // Inclusive scan via shfl_up
    int tmp;
    tmp = __shfl_up_sync(mask, v, 1);  if (tid >= 1)  v += tmp;
    tmp = __shfl_up_sync(mask, v, 2);  if (tid >= 2)  v += tmp;
    tmp = __shfl_up_sync(mask, v, 4);  if (tid >= 4)  v += tmp;
    tmp = __shfl_up_sync(mask, v, 8);  if (tid >= 8)  v += tmp;
    tmp = __shfl_up_sync(mask, v, 16); if (tid >= 16) v += tmp;
    if (tid < n) out[tid] = v;
}
