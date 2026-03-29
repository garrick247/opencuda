// Probe: __device__ __constant__ and texture array patterns
// - __constant__ array accessed through index
// - __constant__ struct
// - __device__ global variable (static storage)
// - Multiple kernels sharing __device__ globals

__device__ int g_counter = 0;
__device__ float g_scale = 1.5f;

struct KernelParams {
    int width;
    int height;
    float scale;
};

__constant__ KernelParams c_params;

__global__ void use_device_global(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = g_counter + tid;
    }
}

__global__ void use_constant_struct(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = tid % (c_params.width * c_params.height);
        out[tid] = in[idx] * c_params.scale * g_scale;
    }
}
