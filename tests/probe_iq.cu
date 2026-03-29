// Probe: complex array index expressions,
// multi-dimensional index computation,
// array of arrays (jagged simulation),
// out-of-order evaluation of complex subscripts

__global__ void complex_index(float *out, float *in, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows * cols) {
        int row = tid / cols;
        int col = tid % cols;
        // Access transposed element
        out[col * rows + row] = in[row * cols + col];
    }
}

// Index by bitfield extraction
__global__ void bitfield_index(float *out, float *table, unsigned int *keys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int k = keys[tid];
        int idx0 = (k >> 0) & 0xFF;
        int idx1 = (k >> 8) & 0xFF;
        int idx2 = (k >> 16) & 0xFF;
        out[tid] = table[idx0] + table[idx1] * 0.5f + table[idx2] * 0.25f;
    }
}

// Negative index (with manual offset to stay in-bounds)
__global__ void center_access(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid > 0 && tid < n - 1) {
        out[tid] = in[tid-1] + in[tid] + in[tid+1];
    } else if (tid == 0 || tid == n-1) {
        out[tid] = in[tid];
    }
}

// Array indexed by function result
__device__ int clamp_idx(int i, int n) {
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

__global__ void func_indexed(float *out, float *in, int *offsets, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int src = clamp_idx(tid + offsets[tid], n);
        out[tid] = in[src];
    }
}
