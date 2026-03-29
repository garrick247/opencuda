// Probe: initializer lists with expressions, conditional compilation guards,
//        multi-dim array access patterns, __launch_bounds__ with 2 args

#define MAX_THREADS 256
#define MIN_BLOCKS 2

__launch_bounds__(MAX_THREADS, MIN_BLOCKS)
__global__ void launch_bounds2(int *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) out[tid] = tid;
}

// Multi-dim array: int arr[4][4] treated as flat
__global__ void flat_matrix(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < 4) {
        int mat[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat[i][j] = i * 4 + j;
            }
        }
        int sum = 0;
        for (int j = 0; j < 4; j++) {
            sum += mat[tid][j];
        }
        out[tid] = sum;
    }
}

// Negative step in for loop
__global__ void reverse_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = n - 1; i >= 0; i--) {
            sum += in[i];
        }
        out[tid] = sum;
    }
}
