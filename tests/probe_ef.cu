// Probe: Patterns specific to device code that differ from host code
// - warpSize (CUDA constant) — should be treated as 32
// - WARP_SIZE macro
// - blockDim used in subscript
// - gridDim in arithmetic expressions
// - Lane ID and warp ID computation

#define FULL_MASK 0xFFFFFFFF

__global__ void warp_scan(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;    // lane within warp
    int warp = tid >> 5;    // warp ID
    
    if (tid < n) {
        int v = in[tid];
        // Inclusive prefix sum within warp using shuffle
        for (int offset = 1; offset < 32; offset <<= 1) {
            int neighbor = __shfl_up_sync(FULL_MASK, v, offset);
            if (lane >= offset) v += neighbor;
        }
        out[tid] = v;
    }
}

// Block-level computation using blockDim
__global__ void block_aware(float *out, float *in, int n) {
    int bdx = blockDim.x;
    int gid = blockIdx.x * bdx + threadIdx.x;
    int stride = gridDim.x * bdx;
    
    float sum = 0.0f;
    for (int i = gid; i < n; i += stride) {
        sum += in[i];
    }
    out[gid < n ? gid : 0] = sum;
}
