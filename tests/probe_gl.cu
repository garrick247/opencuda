// Probe: __device__ function that uses both shared memory and registers,
// inlined into kernel that also has shared memory — verify no collision

__shared__ float g_smem[256];  // module-level extern shared (dynamic)

__device__ float dot_shared(int tid, int n) {
    if (tid < n && tid < 256) {
        return g_smem[tid] * g_smem[(tid + 1) % 256];
    }
    return 0.0f;
}

__global__ void shared_device_fn(float *out, float *in, int n) {
    __shared__ float local_smem[256];
    int tid = threadIdx.x;

    if (tid < n && tid < 256) {
        local_smem[tid] = in[tid];
        g_smem[tid] = in[tid] * 2.0f;
    }
    __syncthreads();

    if (tid < n) {
        float local_dot = local_smem[tid] * local_smem[(tid + 1) % 256];
        float global_dot = dot_shared(tid, n);
        out[tid] = local_dot + global_dot;
    }
}

// Nested shared mem — kernel declares shared, calls device func that
// reads module-level extern shared
__global__ void nested_shared(float *out, float *in, int n) {
    __shared__ float my_buf[128];
    int tid = threadIdx.x;
    if (tid < n && tid < 128) {
        my_buf[tid] = in[tid];
        g_smem[tid] = in[tid] + 1.0f;
    }
    __syncthreads();
    if (tid < 128) {
        out[tid] = my_buf[tid] + dot_shared(tid, n);
    }
}
