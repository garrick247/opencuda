// Probe: global/device/constant memory declarations, extern globals,
// static device variables, and kernel interaction with module-scope data.

// Module-scope global (device memory, writable).
__device__ int g_counter = 0;
__device__ float g_scale = 1.0f;
__device__ unsigned int g_flags = 0xDEADBEEFu;

// Constant array (read-only, broadcast).
__constant__ float c_weights[8] = {0.125f, 0.125f, 0.125f, 0.125f,
                                    0.125f, 0.125f, 0.125f, 0.125f};
__constant__ int c_offsets[4] = {-2, -1, 1, 2};

// ------------------------------------------------------------------
// Read from __device__ global.

__global__ void read_device_global(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * g_scale;
    }
}

// ------------------------------------------------------------------
// Write to __device__ global (atomic increment).

__global__ void write_device_global(int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(&g_counter, 1);
    }
}

// ------------------------------------------------------------------
// Read __constant__ float array.

__global__ void const_weight_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            int src = tid + i - 4;
            if (src >= 0 && src < n) {
                sum += in[src] * c_weights[i];
            }
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Read __constant__ int offsets array.

__global__ void const_offset_access(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = 0;
        for (int i = 0; i < 4; i++) {
            int src = tid + c_offsets[i];
            if (src >= 0 && src < n) {
                v += in[src];
            }
        }
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Read __device__ uint flags with bitwise ops.

__global__ void flag_check(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int f = g_flags;
        int bit = (int)((f >> (tid & 31)) & 1u);
        out[tid] = bit;
    }
}

// ------------------------------------------------------------------
// Multiple globals in one kernel.

__global__ void multi_global(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float scale = g_scale;
        int cnt = g_counter;
        float v = in[tid] * scale + (float)cnt * 0.001f;
        float wsum = 0.0f;
        for (int i = 0; i < 8; i++) {
            int src = tid + i - 4;
            if (src >= 0 && src < n) {
                wsum += in[src] * c_weights[i];
            }
        }
        out[tid] = v + wsum;
    }
}
