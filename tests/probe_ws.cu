// Probe: multi-block reduction with atomic aggregation, prefix-sum
// with shared memory, scan patterns, and pointer-to-pointer patterns.

// ------------------------------------------------------------------
// Block-level sum reduction using shared memory tree.

__global__ void block_reduce_sum(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}

// ------------------------------------------------------------------
// Inclusive prefix sum (scan) within a block.

__global__ void block_scan(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Hillis-Steele scan
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = (tid >= offset) ? smem[tid - offset] : 0;
        __syncthreads();
        smem[tid] += val;
        __syncthreads();
    }

    if (gid < n) {
        out[gid] = smem[tid];
    }
}

// ------------------------------------------------------------------
// Pointer-to-pointer: array of pointers (simulated via index table).

__global__ void indirect_gather(int *out, int *data, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid];
        out[tid] = data[idx];
    }
}

// ------------------------------------------------------------------
// Scatter: write to indexed position.

__global__ void indirect_scatter(int *out, int *data, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int dst = indices[tid];
        out[dst] = data[tid];
    }
}

// ------------------------------------------------------------------
// 2D matrix transpose using shared memory.

__global__ void matrix_transpose(float *out, float *in, int rows, int cols) {
    __shared__ float tile[16][16];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int r = by * 16 + ty;
    int c = bx * 16 + tx;

    if (r < rows && c < cols) {
        tile[ty][tx] = in[r * cols + c];
    }
    __syncthreads();

    int tr = bx * 16 + ty;
    int tc = by * 16 + tx;
    if (tr < cols && tc < rows) {
        out[tr * rows + tc] = tile[tx][ty];
    }
}

// ------------------------------------------------------------------
// Parallel histogram with atomic adds.

__global__ void histogram_atomic(int *hist, int *in, int n, int nbins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int bin = in[tid] % nbins;
        atomicAdd(&hist[bin], 1);
    }
}

// ------------------------------------------------------------------
// Stream compaction: mark-then-compact phase 1 (mark valid elements).

__global__ void mark_valid(int *mask, int *in, int n, int threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        mask[tid] = (in[tid] > threshold) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Prefix product (not sum): multiply all previous.

__global__ void prefix_product(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;

    smem[tid] = (tid < n) ? in[tid] : 1;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = (tid >= offset) ? smem[tid - offset] : 1;
        __syncthreads();
        smem[tid] *= val;
        __syncthreads();
    }

    if (tid < n) {
        out[tid] = smem[tid];
    }
}

// ------------------------------------------------------------------
// Stencil computation: 1D 3-point stencil.

__global__ void stencil_1d(float *out, float *in, int n) {
    __shared__ float smem[258];   // 256 + 2 halo
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load interior
    smem[tid + 1] = (gid < n) ? in[gid] : 0.0f;

    // Load halo
    if (tid == 0 && gid > 0)
        smem[0] = in[gid - 1];
    else if (tid == 0)
        smem[0] = 0.0f;

    if (tid == blockDim.x - 1 && gid + 1 < n)
        smem[tid + 2] = in[gid + 1];
    else if (tid == blockDim.x - 1)
        smem[tid + 2] = 0.0f;

    __syncthreads();

    if (gid > 0 && gid < n - 1) {
        out[gid] = 0.25f * smem[tid] + 0.5f * smem[tid + 1] + 0.25f * smem[tid + 2];
    }
}

// ------------------------------------------------------------------
// Running max with reset: conditional accumulation.

__global__ void running_max(int *out, int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int cur = 0;
        // Find max over rolling window of 4 (prev 3 + self)
        int start = (tid >= 3) ? tid - 3 : 0;
        for (int i = start; i <= tid; i++) {
            int x = in[i];
            if (x > cur) cur = x;
        }
        out[tid] = cur;
    }
}
