// Probe: volatile memory, __threadfence family, restrict pointers,
// memory fence intrinsics, and mixed address-space access patterns.

// ------------------------------------------------------------------
// Volatile load/store: must not be cached or reordered.

__global__ void volatile_rw(volatile int *out, volatile int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = v * 2;
    }
}

// ------------------------------------------------------------------
// __threadfence: system-level memory fence.

__global__ void fence_write(int *flag, int *data, int val) {
    int tid = threadIdx.x;
    if (tid == 0) {
        data[0] = val;
        __threadfence();
        flag[0] = 1;
    }
}

// ------------------------------------------------------------------
// __threadfence_block: block-level fence.

__global__ void fence_block(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = in[tid] + 1;
    }
    __threadfence_block();
    if (tid < n) {
        out[tid] = smem[(tid + 1) % n];
    }
}

// ------------------------------------------------------------------
// __threadfence_system: cross-device fence (maps to .sys).

__global__ void fence_system(int *flag) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        flag[0] = 1;
    }
}

// ------------------------------------------------------------------
// restrict pointers: compiler hint, no semantic difference in PTX.

__global__ void restrict_add(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// const restrict: read-only input with restrict.

__global__ void scale_restrict(float * __restrict__ out,
                                const float * __restrict__ in,
                                float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * scale;
    }
}

// ------------------------------------------------------------------
// Multiple fence types in same kernel.

__global__ void mixed_fences(int *out, int *in, int n) {
    __shared__ int buf[256];
    int tid = threadIdx.x;
    if (tid < n) {
        buf[tid] = in[tid];
    }
    __threadfence_block();
    if (tid < n) {
        int v = buf[tid];
        out[tid] = v;
        __threadfence();
    }
}

// ------------------------------------------------------------------
// Volatile in shared memory.

__global__ void volatile_shared(int *out, int *in, int n) {
    __shared__ volatile int smem[256];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = in[tid];
        __syncthreads();
        out[tid] = smem[n - 1 - tid];
    }
}
