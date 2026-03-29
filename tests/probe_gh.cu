// Probe: unusual array patterns — VLA (variable-length arrays, just parse),
// zero-length array member in struct (extension, parse-only),
// array decay to pointer, sizeof on array vs pointer

// Array passed by pointer — inside function it decays
__device__ void fill_arr(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) {
        arr[i] = val;
    }
}

// Stack array of known size, then passed as pointer
__global__ void local_array_decay(float *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float local[8];
        fill_arr(local, 8, (float)tid);
        for (int i = 0; i < 8; i++) {
            out[i] = local[i];
        }
    }
}

// Multidimensional indexing via 1D array
__global__ void flat_2d(float *out, float *in, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows * cols) {
        int row = tid / cols;
        int col = tid % cols;
        // Transpose
        out[col * rows + row] = in[tid];
    }
}

// Stride-based access pattern (as if iterating over a 2D subview)
__global__ void strided_copy(float *out, float *in, int n, int src_stride, int dst_stride) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid * dst_stride] = in[tid * src_stride];
    }
}
