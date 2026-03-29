// Regression: 2D __shared__ memory indexing: tile[ty][tx] = ...
// Without fix: tile[ty][tx] → ParseError "expected SEMI, got LBRACKET '['"
//   because tile[ty] computed stride=sizeof(float)=4 (treated tile as 1D
//   and returned the loaded scalar), leaving [tx] as an unexpected token.
// Fix: parser tracks row strides for multi-dim __shared__ arrays;
//   tile[ty] uses row_stride=16*4=64, returns pointer (not loaded value),
//   then tile[ty][tx] computes addr = tile + ty*64 + tx*4.

__global__ void transpose(float *out, float *in, int N) {
    __shared__ float tile[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 2D read from global, 2D write to shared
    tile[ty][tx] = in[ty * N + tx];
    __syncthreads();

    // 2D read from shared (transposed), write to global
    out[tx * N + ty] = tile[tx][ty];
}

__global__ void matmul_tile(float *C, float *A, float *B, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float sum = 0.0f;
    int k;
    for (k = 0; k < N; k += 16) {
        As[ty][tx] = A[row * N + k + tx];
        Bs[ty][tx] = B[(k + ty) * N + col];
        __syncthreads();
        int t;
        for (t = 0; t < 16; t++) {
            sum += As[ty][t] * Bs[t][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

__global__ void smem_int2d(int *out, int *in, int N) {
    __shared__ int buf[8][8];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx < 8 && ty < 8) {
        buf[ty][tx] = in[ty * N + tx];
        __syncthreads();
        out[ty * N + tx] = buf[ty][tx] + buf[tx][ty];
    }
}
