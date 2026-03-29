// Probe: const __device__ lookup tables, switch fall-through patterns,
// complex nested && || chains, and 2D/3D pointer-index arithmetic.

// ------------------------------------------------------------------
// const __device__ lookup table (read-only).

__device__ const float G_SIN_LUT[8] = {
    0.0f, 0.7071f, 1.0f, 0.7071f, 0.0f, -0.7071f, -1.0f, -0.7071f
};

__global__ void apply_lut(float *out, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid] & 7;
        out[tid] = G_SIN_LUT[idx];
    }
}

// ------------------------------------------------------------------
// Switch with multiple cases and fall-through (explicit break).

__global__ void switch_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r;
        switch (v % 5) {
            case 0: r = 100; break;
            case 1: r = 200; break;
            case 2: r = 300; break;
            case 3: r = 400; break;
            default: r = 999; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Complex nested && / || chains.

__global__ void complex_cond(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid], z = c[tid];
        // ((x > 0 && y > 0) || z > 10) && (x + y < 100)
        int cond = ((x > 0 && y > 0) || z > 10) && (x + y < 100);
        out[tid] = cond ? x + y + z : 0;
    }
}

// ------------------------------------------------------------------
// 2D array index arithmetic: row-major A[i][j] = A_flat[i*cols + j].

__global__ void matmul_1d(float *C, float *A, float *B, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ------------------------------------------------------------------
// Pointer-to-pointer (int **): indirect array access.

__global__ void indirect_access(int *out, int **ptrs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *row = ptrs[tid];
        out[tid] = row[0] + row[1];
    }
}

// ------------------------------------------------------------------
// Long integer arithmetic: 64-bit multiply and divide.

__global__ void long_arith(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long x = a[tid];
        long long y = b[tid];
        out[tid] = x * y + (x / (y != 0 ? y : 1));
    }
}

// ------------------------------------------------------------------
// Short-circuit: null pointer guard before dereference.

__global__ void null_guard(int *out, int **ptrs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = ptrs[tid];
        out[tid] = (p != 0 && p[0] > 0) ? p[0] : -1;
    }
}
