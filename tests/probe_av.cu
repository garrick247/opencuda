// Probe: complex real-world patterns
// - __shared__ array accessed with stride
// - warp-level reduction with __shfl_down_sync
// - multiple __shared__ arrays in same kernel
// - __syncthreads() barrier usage

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void warp_reduce_sum(float *out, float *in, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Block reduction
    for (int stride = BLOCK_SIZE / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Warp reduction
    if (tid < WARP_SIZE) {
        float val = smem[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) out[blockIdx.x] = val;
    }
}

__global__ void two_pass_normalize(float *out, float *in, int n) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sq[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    float v = (gid < n) ? in[gid] : 0.0f;
    s_sum[tid] = v;
    s_sq[tid] = v * v;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid]  += s_sq[tid + stride];
        }
        __syncthreads();
    }

    if (gid < n) {
        float mean = s_sum[0] / (float)n;
        float var  = s_sq[0] / (float)n - mean * mean;
        float inv_std = 1.0f / sqrtf(var + 1e-6f);
        out[gid] = (in[gid] - mean) * inv_std;
    }
}
