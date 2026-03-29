// Probe: pointer compound assignment (+= / -=),
// pointer comparison for loop termination,
// pointer iteration pattern (p++ as loop advance),
// pointer difference used as count

// Pointer walk via p++ in loop
__global__ void sum_via_ptr(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        float *p = in;
        float *end = in + n;   // pointer one past the last element
        while (p < end) {
            sum += *p;
            p++;
        }
        *out = sum;
    }
}

// Pointer += stride
__global__ void stride_sum_ptr(float *out, float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        float *p = in;
        float *end = in + n;
        while (p < end) {
            sum += *p;
            p += stride;   // compound pointer advance
        }
        *out = sum;
    }
}

// Row-pointer pattern: ptr points to start of each row
__global__ void row_sums(float *out, float *matrix, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        float *row = matrix + tid * cols;
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += row[j];
        }
        out[tid] = sum;
    }
}

// Pointer arithmetic in expression: *(ptr + i) vs ptr[i]
__global__ void ptr_vs_subscript(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // *(in + tid) should equal in[tid]
        out[tid] = *(in + tid);
    }
}
