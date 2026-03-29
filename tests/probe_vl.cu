// Probe: __device__ global array access, device function forward declarations,
// printf with varied format specifiers.

// ------------------------------------------------------------------
// __device__ global variable (scalar).

__device__ int g_counter;

__global__ void global_counter_inc(int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(&g_counter, 1);
    }
}

// ------------------------------------------------------------------
// __device__ global array.

__device__ float g_lut[8];

__global__ void global_array_write(float *values, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        g_lut[tid] = values[tid];
    }
}

__global__ void global_array_read(float *out, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid] & 7;  // ensure in bounds
        out[tid] = g_lut[idx];
    }
}

// ------------------------------------------------------------------
// Forward declaration of device function.

__device__ int fwd_compute(int v);  // forward decl

__global__ void uses_fwd(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fwd_compute(in[tid]);
    }
}

__device__ int fwd_compute(int v) {  // definition after use
    return v * v + v + 1;
}

// ------------------------------------------------------------------
// Printf with various format specifiers.

__global__ void printf_formats(int *in, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && tid < n) {
        int iv = in[tid];
        float fv = fin[tid];
        // Test various format codes
        printf("int=%d float=%.2f uint=%u\n", iv, fv, (unsigned)iv);
    }
}

// ------------------------------------------------------------------
// printf from multiple threads (tid < 4).

__global__ void printf_multi_thread(int *in, int n) {
    int tid = threadIdx.x;
    if (tid < 4 && tid < n) {
        printf("thread %d value %d\n", tid, in[tid]);
    }
}

// ------------------------------------------------------------------
// __device__ function referencing a global.

__device__ int g_scale = 3;

__device__ int scale_val(int v) {
    return v * g_scale;
}

__global__ void use_global_device(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = scale_val(in[tid]);
    }
}
