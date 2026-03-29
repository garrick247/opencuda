// Probe: complex reduction patterns — tree reduction in shared mem,
// warp-level then block-level reduction

#define BLOCK_SIZE 256

__global__ void block_sum_reduce(float *out, float *in, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load
    sdata[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// Warp reduce then write once per warp
__global__ void warp_sum_output(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float val = (gid < n) ? in[gid] : 0.0f;

    // Warp reduction via shfl_down
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        out[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}
