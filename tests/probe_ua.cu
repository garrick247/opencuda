// Probe: real-world GPU algorithm patterns — image processing,
// graph algorithms, and text processing patterns.

// ------------------------------------------------------------------
// Box filter (2D convolution with constant kernel).

__global__ void box_filter(float *out, const float *in, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        float sum = 0.0f;
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    sum += in[ny * W + nx];
                    count++;
                }
            }
        }
        out[y * W + x] = sum / (float)count;
    }
}

// ------------------------------------------------------------------
// Prefix sum (scan) — single block version.

__global__ void prefix_sum(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    smem[tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        int val = 0;
        if (tid >= stride) val = smem[tid - stride];
        __syncthreads();
        smem[tid] += val;
    }
    __syncthreads();
    if (tid < n) out[tid] = smem[tid];
}

// ------------------------------------------------------------------
// Counting sort (frequency count).

__global__ void count_freq(int *freq, const int *in, int n, int max_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v >= 0 && v < max_val) {
            atomicAdd(&freq[v], 1);
        }
    }
}

// ------------------------------------------------------------------
// Sparse vector dot product.

__global__ void sparse_dot(float *out, const int *idx_a, const float *val_a, int nnz_a,
                             const float *dense, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int i = 0; i < nnz_a; i++) {
            int j = idx_a[i];
            if (j < n) {
                acc += val_a[i] * dense[j];
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Run-length encoding detection (find run boundaries).

__global__ void rle_detect(int *boundaries, const int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (tid == 0) {
            boundaries[tid] = 1;
        } else {
            boundaries[tid] = (in[tid] != in[tid - 1]) ? 1 : 0;
        }
    }
}

// ------------------------------------------------------------------
// Weighted moving average (FIR filter).

__global__ void fir_filter(float *out, const float *in,
                             const float *coeffs, int n, int taps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int k = 0; k < taps; k++) {
            int src = tid - k;
            if (src >= 0) {
                acc += coeffs[k] * in[src];
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Bitonic sort step (single step).

__global__ void bitonic_step(int *data, int n, int j, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int l = tid ^ j;
        if (l > tid) {
            int ascending = ((tid & k) == 0);
            int a = data[tid], b = data[l];
            if ((ascending && a > b) || (!ascending && a < b)) {
                data[tid] = b;
                data[l]   = a;
            }
        }
    }
}
