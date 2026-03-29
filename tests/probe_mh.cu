// Probe: Cooperative memory patterns
// - Warp-level reduction using __shfl_down_sync
// - __syncthreads inside conditional (thread divergence)
// - atomicAdd with struct-derived address
// - __shared__ struct array usage
// - Histogram with atomicAdd

__global__ void warp_reduce_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    float val = (tid < n) ? in[tid] : 0.0f;
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (lane == 0) {
        atomicAdd(out, val);
    }
}

// __shared__ struct array
struct Accum {
    float sum;
    int count;
};

__global__ void shared_struct(float *out, float *in, int n) {
    __shared__ float s_sum[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    s_sum[lane] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    // Simple sequential reduction in lane 0
    if (lane == 0) {
        float total = 0.0f;
        for (int i = 0; i < 32; i++) {
            total += s_sum[i];
        }
        out[blockIdx.x] = total;
    }
}

// atomicAdd with computed address
__global__ void histogram(int *hist, int *data, int n, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int bin = data[tid] % num_bins;
        atomicAdd(&hist[bin], 1);
    }
}

// Thread divergence with syncthreads
__global__ void conditional_sync(float *out, float *in, int n, float threshold) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    if (tid < n) {
        shared[tid] = in[tid];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();
    if (tid < n) {
        float v = shared[tid];
        if (v > threshold) {
            out[tid] = v * 2.0f;
        } else {
            out[tid] = v * 0.5f;
        }
    }
}

// Mixed warp/block ops
__global__ void prefix_sum(int *out, int *in, int n) {
    __shared__ int s_data[1024];
    int tid = threadIdx.x;
    s_data[tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();
    // Up-sweep (reduce phase)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            s_data[index] += s_data[index - stride];
        }
        __syncthreads();
    }
    if (tid < n) {
        out[tid] = s_data[tid];
    }
}
