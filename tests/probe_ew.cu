// Probe: global __device__ variables (not kernels), extern __device__,
// __device__ variable reads inside kernels

__device__ int g_counter;
__device__ float g_scale = 2.0f;

__global__ void read_device_var(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * g_scale;
    }
}

// __device__ array variable
__device__ float g_weights[4] = {0.25f, 0.25f, 0.25f, 0.25f};

__global__ void weighted_avg(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            int idx = (tid + i) % n;
            sum += in[idx] * g_weights[i];
        }
        out[tid] = sum;
    }
}

// Atomic add to __device__ global counter
__global__ void count_positive(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && in[tid] > 0) {
        atomicAdd(&g_counter, 1);
    }
    if (tid == 0) {
        out[0] = g_counter;
    }
}
