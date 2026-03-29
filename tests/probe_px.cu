// Probe: global __device__ variables — arrays, structs, read/write,
// atomics on globals, and multiple kernels sharing globals.

// ------------------------------------------------------------------
// Global __device__ array: read-only lookup table.

__device__ int g_lut[8] = {0, 1, 4, 9, 16, 25, 36, 49};

__global__ void lut_lookup(int *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 7;  // clamp to [0,7]
        out[tid] = g_lut[i];
    }
}

// ------------------------------------------------------------------
// Global __device__ scalar: read + write (two separate kernels).
// Kernel A increments, kernel B reads.

__device__ int g_counter = 0;

__global__ void increment_global(int count) {
    int tid = threadIdx.x;
    if (tid == 0) {
        atomicAdd(&g_counter, count);
    }
}

__global__ void read_global(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = g_counter;
    }
}

// ------------------------------------------------------------------
// Global __device__ float array: scale factors per channel.

__device__ float g_scales[4] = {1.0f, 2.0f, 0.5f, 0.25f};

__global__ void apply_scales(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int ch = tid % 4;
        out[tid] = data[tid] * g_scales[ch];
    }
}

// ------------------------------------------------------------------
// Global __device__ struct: single configuration struct.

struct Config {
    int   width;
    int   height;
    float scale;
};

__device__ Config g_config;

__global__ void use_config(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Read config fields
        int w = g_config.width;
        float s = g_config.scale;
        int row = tid / w;
        out[tid] = data[tid] * s * (float)(row + 1);
    }
}

// ------------------------------------------------------------------
// Atomic operations on global array elements.
// Multiple threads write to per-bin counters.

__device__ int g_bins[16];

__global__ void histogram_global(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int bin = data[tid] & 15;
        atomicAdd(&g_bins[bin], 1);
    }
}

__global__ void read_histogram(int *out) {
    int tid = threadIdx.x;
    if (tid < 16) {
        out[tid] = g_bins[tid];
    }
}

// ------------------------------------------------------------------
// Two kernels share the same __device__ fn and global.
// Tests that multiple kernels in one PTX can both reference the global.

__device__ int g_bias = 7;

__device__ int apply_bias(int x) {
    return x + g_bias;
}

__global__ void bias_kernel_a(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = apply_bias(data[tid]);
    }
}

__global__ void bias_kernel_b(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = apply_bias(data[tid]) * 2;
    }
}
