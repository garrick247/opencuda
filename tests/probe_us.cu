// Probe: real-world GPU algorithm patterns — image histogram,
// parallel prefix sum variants, radix sort step, and merge-path.

// ------------------------------------------------------------------
// 2D histogram with shared memory accumulation.

__global__ void histogram2d(int *hist, const unsigned char *img,
                              int W, int H, int bins) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x;

    // Init local histogram
    if (tid < bins) local_hist[tid] = 0;
    __syncthreads();

    // Each thread processes a pixel
    int gid = blockIdx.x * blockDim.x + tid;
    int total = W * H;
    while (gid < total) {
        int val = (int)img[gid];
        if (val < bins) {
            atomicAdd(&local_hist[val], 1);
        }
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();

    // Merge local into global
    if (tid < bins) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

// ------------------------------------------------------------------
// Exclusive prefix sum (scan) using Blelloch algorithm.

__global__ void blelloch_scan(int *data, int n) {
    __shared__ int smem[512];
    int tid = threadIdx.x;

    smem[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    int stride = 1;
    while (stride < blockDim.x) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            smem[idx] += smem[idx - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    // Clear root
    if (tid == 0) smem[blockDim.x - 1] = 0;
    __syncthreads();

    // Down-sweep
    stride = blockDim.x / 2;
    while (stride > 0) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            int tmp = smem[idx - stride];
            smem[idx - stride] = smem[idx];
            smem[idx] += tmp;
        }
        stride /= 2;
        __syncthreads();
    }

    if (tid < n) data[tid] = smem[tid];
}

// ------------------------------------------------------------------
// Radix sort digit extraction (1 pass of 8-bit radix sort).

__global__ void radix_count(int *count, const unsigned int *keys, int n, int shift) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int digit = (keys[gid] >> shift) & 0xFF;
        atomicAdd(&count[digit], 1);
    }
}

// ------------------------------------------------------------------
// Stream compaction: compact positive elements.

__global__ void compact_positive(int *out, int *count_out,
                                   const int *in, int n) {
    __shared__ int block_out[256];
    __shared__ int block_cnt;
    int tid = threadIdx.x;

    if (tid == 0) block_cnt = 0;
    __syncthreads();

    int val = (tid < n) ? in[tid] : 0;
    int pos = -1;
    if (tid < n && val > 0) {
        pos = atomicAdd(&block_cnt, 1);
    }
    __syncthreads();

    if (pos >= 0) block_out[pos] = val;
    __syncthreads();

    if (tid < block_cnt) {
        int write_base = (tid == 0) ? atomicAdd(count_out, block_cnt) : 0;
        (void)write_base;  // used by first thread only for global offset
        out[tid] = block_out[tid];
    }
}

// ------------------------------------------------------------------
// Segmented reduction by key (value sorted, reduce within key groups).

__global__ void segmented_reduce(float *out, const float *in,
                                   const int *keys, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float acc = in[gid];
        int key = keys[gid];
        // Look forward and accumulate while key matches
        // (simplified: just output value + count of matching neighbors)
        int cnt = 1;
        // Look right
        int j = gid + 1;
        while (j < n && keys[j] == key && j < gid + 8) {
            acc += in[j];
            cnt++;
            j++;
        }
        out[gid] = acc / (float)cnt;
    }
}
