// Probe: Edge cases in __shared__ memory usage
// - __shared__ 2D array: float smem[16][16]
// - __shared__ struct array: MyStruct smem[32]
// - __shared__ variable with initializer (invalid but should parse)
// - Multiple __shared__ decls in same kernel
// - extern __shared__ after other shared decls

struct Tile {
    float data[8];
};

__global__ void shared_2d(float *out, float *in, int n) {
    __shared__ float smem[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Load
    if (tx < 16 && ty < 16) {
        int idx = ty * 16 + tx;
        smem[ty][tx] = (idx < n) ? in[idx] : 0.0f;
    }
    __syncthreads();
    // Write back transposed
    if (tx < 16 && ty < 16) {
        int idx = tx * 16 + ty;
        if (idx < n) out[idx] = smem[tx][ty];
    }
}

__global__ void shared_multi(float *out, float *a, float *b, int n) {
    __shared__ float sa[128];
    __shared__ float sb[128];
    int tid = threadIdx.x;
    if (tid < 128) {
        sa[tid] = (tid < n) ? a[tid] : 0.0f;
        sb[tid] = (tid < n) ? b[tid] : 0.0f;
    }
    __syncthreads();
    if (tid < n && tid < 128) {
        out[tid] = sa[tid] + sb[tid];
    }
}
