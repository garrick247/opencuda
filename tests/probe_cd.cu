// Probe: Unusual but valid variable naming
// - Variables named with trailing underscores: val_, ptr_
// - Variables starting with double underscore (reserved in C++ but parseable)
// - Very deeply nested expression: a + b + c + d + e + f + g + h

__global__ void deep_expr(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid];
        float b = a * 1.1f;
        float c = b + 0.5f;
        float d = c - a;
        float e = d * b;
        float f2 = e + c;
        float g = f2 * d;
        float h = g - e;
        // Deeply nested add chain
        out[tid] = a + b + c + d + e + f2 + g + h;
    }
}

__global__ void underscore_names(float *out_, float *in_, int n_) {
    int tid_ = threadIdx.x;
    if (tid_ < n_) {
        float val_ = in_[tid_];
        float result_ = val_ * val_;
        out_[tid_] = result_;
    }
}

// Test: index computation with many intermediate values
__global__ void index_math(int *out, int W, int H, int stride) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int gx = tx + bx * blockDim.x;
    int gy = ty + by * blockDim.y;
    if (gx < W && gy < H) {
        int row_idx = gy * stride;
        int col_idx = gx;
        int flat = row_idx + col_idx;
        out[flat] = flat;
    }
}
