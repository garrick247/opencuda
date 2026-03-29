// Probe: warp-level vote/match/reduce ops, abs/fabs, integer min/max
// with different types, and mixed global/shared memory access patterns.

// ------------------------------------------------------------------
// Warp voting operations.

__global__ void warp_vote_ops(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        unsigned mask = 0xFFFFFFFF;

        // Vote: all/any across warp
        int all_pos = __all_sync(mask, v > 0);
        int any_neg = __any_sync(mask, v < 0);
        int ballot = (int)__ballot_sync(mask, v > 100);

        out[gid * 3]     = all_pos;
        out[gid * 3 + 1] = any_neg;
        out[gid * 3 + 2] = ballot;
    }
}

// ------------------------------------------------------------------
// abs/fabs on various types.

__global__ void abs_ops(int *iout, float *fout, double *dout,
                          int *iin, float *fin, double *din, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        iout[tid] = abs(iin[tid]);
        fout[tid] = fabsf(fin[tid]);
        dout[tid] = fabs(din[tid]);
    }
}

// ------------------------------------------------------------------
// min/max on various types.

__global__ void minmax_ops(int *iout, float *fout,
                             int *iin, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int iv = iin[tid];
        float fv = fin[tid];

        iout[tid * 4 + 0] = min(iv, 0);
        iout[tid * 4 + 1] = max(iv, 0);
        iout[tid * 4 + 2] = min(iv, 100);
        iout[tid * 4 + 3] = max(iv, -100);

        fout[tid * 4 + 0] = fminf(fv, 0.0f);
        fout[tid * 4 + 1] = fmaxf(fv, 0.0f);
        fout[tid * 4 + 2] = fminf(fv, 1.0f);
        fout[tid * 4 + 3] = fmaxf(fv, -1.0f);
    }
}

// ------------------------------------------------------------------
// Mixed shared + global access in same kernel.

__global__ void mixed_shared_global(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load from global into shared
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Compute using both shared (neighbor) and global (self)
    float self = (gid < n) ? in[gid] : 0.0f;  // re-read from global
    float neighbor = smem[(tid + 1) % blockDim.x];

    if (gid < n) {
        out[gid] = self * 0.5f + neighbor * 0.5f;
    }
}

// ------------------------------------------------------------------
// Warp shuffle sum reduction.

__global__ void warp_shuffle_sum(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < n) ? in[gid] : 0.0f;

    unsigned mask = __activemask();
    // Butterfly reduction within warp
    val += __shfl_xor_sync(mask, val, 16);
    val += __shfl_xor_sync(mask, val, 8);
    val += __shfl_xor_sync(mask, val, 4);
    val += __shfl_xor_sync(mask, val, 2);
    val += __shfl_xor_sync(mask, val, 1);

    // Lane 0 writes result
    if ((threadIdx.x & 31) == 0) {
        out[gid >> 5] = val;
    }
}

// ------------------------------------------------------------------
// Global memory coalescing pattern: transposed read.

__global__ void transpose_read(float *out, float *in, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        // Transposed: write to [c * rows + r] from [r * cols + c]
        out[c * rows + r] = in[r * cols + c];
    }
}
