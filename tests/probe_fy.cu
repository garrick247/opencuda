// Probe: Extremely dense compute kernel — softmax implementation
// Exercises: exp, log, atomics, shared memory, warp shuffles, complex control flow

#define WARP_SIZE 32

__global__ void softmax_warp(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid >= n) return;

    float val = in[gid];

    // Warp max reduction
    float warp_max = val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
        if (other > warp_max) warp_max = other;
    }
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);  // broadcast max

    // Compute exp(val - max)
    float exp_val = __expf(val - warp_max);

    // Warp sum of exp
    float warp_sum = exp_val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }
    warp_sum = __shfl_sync(0xFFFFFFFF, warp_sum, 0);

    out[gid] = exp_val / warp_sum;
}

__global__ void layer_norm(float *out, float *in, float *gamma, float *beta,
                             int n, float eps) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < n) ? in[gid] : 0.0f;
    smem[tid] = val;
    __syncthreads();

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        mean += smem[i];
    }
    mean /= (float)blockDim.x;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        float diff = smem[i] - mean;
        var += diff * diff;
    }
    var /= (float)blockDim.x;

    if (gid < n) {
        float norm = (val - mean) / sqrtf(var + eps);
        out[gid] = gamma[tid] * norm + beta[tid];
    }
}
