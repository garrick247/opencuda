// Probe: blockIdx/blockDim for global thread ID,
// 2D thread indexing (threadIdx.y, blockDim.y),
// two separate for-loops in if-else branches modifying same outer var,
// complex use of loop variable in both branches of if-else inside loop

// Global thread ID with blockIdx.x * blockDim.x + threadIdx.x
__global__ void global_tid(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = in[gid] * 2;
    }
}

// 2D thread block: row = threadIdx.y, col = threadIdx.x
__global__ void thread2d(int *out, int *in, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[row * cols + col] = in[row * cols + col];
    }
}

// Two loops in separate branches of if-else, both modify 'result'
__global__ void two_loops_branches(int *out, int *in, int n, int mode) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int result = 0;
        if (mode == 0) {
            for (int i = 0; i < n; i++) {
                result += in[i];          // sum
            }
        } else {
            for (int i = 0; i < n; i++) {
                result += in[i] * in[i];  // sum of squares
            }
        }
        *out = result;
    }
}

// If-else inside loop: loop var used in both branches with different expressions
__global__ void if_in_loop_both_branches(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int even_sum = 0;
        int odd_sum = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                even_sum += in[i];    // uses i in parity check, in[i] for value
            } else {
                odd_sum += in[i];
            }
        }
        out[0] = even_sum;
        out[1] = odd_sum;
    }
}

// blockIdx.x used as input to kernel logic (not just tid computation)
__global__ void block_indexed_work(int *out, int *in, int items_per_block) {
    int block_start = blockIdx.x * items_per_block;
    int tid = threadIdx.x;
    int global_idx = block_start + tid;
    out[global_idx] = in[global_idx] + blockIdx.x;  // add block ID as offset
}
