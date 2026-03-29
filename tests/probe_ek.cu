// Probe: multi-dimensional array access (2D, 3D), row-major index computation
// Also: array of structs vs struct of arrays pattern

struct Pair {
    float x, y;
};

__global__ void matrix2d(float *out, float *in, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += in[tid * cols + j];
        }
        out[tid] = sum;
    }
}

// Array-of-structs access pattern
__global__ void aos_access(float *out, Pair *pairs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = pairs[tid].x + pairs[tid].y;
    }
}

// Nested loop with 2D-style indexing
__global__ void block_reduce(float *out, float *in, int rows, int cols) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid < rows && tid < cols) {
        float val = in[bid * cols + tid];
        // Simple per-thread output
        out[bid * cols + tid] = val * val;
    }
}
