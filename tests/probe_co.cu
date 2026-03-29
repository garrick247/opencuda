// Probe: Non-trivial use of __shared__ with synchronization
// - Parallel prefix sum (scan) algorithm
// - Double-buffered tiling
// - Warp-level primitives mixed with shared memory

#define BLOCK_SIZE 256

__global__ void prefix_sum_block(int *out, int *in, int n) {
    __shared__ int smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    
    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < BLOCK_SIZE) {
            smem[idx] += smem[idx - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx + stride < BLOCK_SIZE) {
            smem[idx + stride] += smem[idx];
        }
        __syncthreads();
    }
    
    if (gid < n) out[gid] = smem[tid];
}
