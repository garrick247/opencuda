// Probe: 2D local arrays, bitwise NOT on various types,
// negation of unsigned (wraps), const-qualified pointer parameter,
// multiple dimensions of blockIdx/blockDim usage

// 2D local array (stack-allocated matrix)
__global__ void local_2d(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mat[3][3];
        // Fill with row*3+col
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                mat[r][c] = r * 3 + c;
            }
        }
        // Sum diagonal
        int diag = mat[0][0] + mat[1][1] + mat[2][2];  // 0+4+8=12
        out[0] = diag;
    }
}

// Bitwise NOT
__global__ void bitwise_not(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ~in[tid];
    }
}

// Bitwise NOT of unsigned
__global__ void bitwise_not_uint(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ~in[tid];
    }
}

// Unsigned negation (wraps: -0u = 0, -1u = 0xFFFFFFFF)
__global__ void unsigned_negate(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (unsigned int)(-(int)in[tid]);
    }
}

// Multi-dimensional grid: uses blockIdx.x, blockIdx.y, blockDim.x, blockDim.y
__global__ void grid_2d(int *out, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;
    out[idx] = row * 1000 + col;
}

// const pointer parameter (read-only)
__global__ void const_ptr(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Just read, don't write through in
        out[tid] = in[tid] + in[n - 1 - tid];
    }
}
