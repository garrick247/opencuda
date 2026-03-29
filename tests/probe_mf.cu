// Probe: Pointer arithmetic and multi-dimensional array access
// - 2D array via flat pointer: a[row * cols + col]
// - Stride-1 vs stride-N access patterns
// - Pointer to pointer-arithmetic result stored in local
// - Base pointer advanced in a loop (pointer bumping)
// - Negative index offset (ptr - k)

__global__ void flat_2d_access(float *out, float *in, int rows, int cols) {
    int tid = threadIdx.x;
    int row = tid / cols;
    int col = tid % cols;
    if (row < rows && col < cols) {
        // in[row][col] = in[row*cols + col]
        float val = in[row * cols + col];
        // transpose: out[col][row] = out[col*rows + row]
        out[col * rows + row] = val;
    }
}

__global__ void stride_n_access(float *out, float *in, int n, int stride) {
    int tid = threadIdx.x;
    if (tid < n) {
        // gather: out[i] = in[i * stride]
        out[tid] = in[tid * stride];
    }
}

// Pointer bumping in a loop
__global__ void pointer_bump(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int *p = in + tid;  // start at in[tid]
        for (int i = 0; i < n; i++) {
            sum += *p;
            p += n;         // stride by n each iteration
            if (p >= in + n * n) break;
        }
        out[tid] = sum;
    }
}

// Negative offset
__global__ void neg_offset(int *out, int *in, int n) {
    int tid = threadIdx.x;
    // skip first element: start at in[1]
    if (tid > 0 && tid < n) {
        int cur = in[tid];
        int prev = *(in + tid - 1);   // in[tid - 1]
        out[tid] = cur - prev;
    } else if (tid == 0) {
        out[0] = in[0];
    }
}

// Pointer stored in local var then used later
__global__ void local_ptr_store(int *out, int *a, int *b, int n, int flag) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *src;
        if (flag) {
            src = a;
        } else {
            src = b;
        }
        out[tid] = src[tid];
    }
}
