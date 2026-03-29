// Probe: memory semantics — volatile, __restrict__, __ldg, memory fences,
// clock(), and special memory access patterns.

// ------------------------------------------------------------------
// Volatile __device__ variable: write and read back.

__device__ volatile int g_volatile_flag;
__device__ volatile float g_volatile_val;

__global__ void set_volatile(int flag, float val) {
    if (threadIdx.x == 0) {
        g_volatile_flag = flag;
        g_volatile_val  = val;
    }
}

__global__ void read_volatile(int *out_flag, float *out_val) {
    if (threadIdx.x == 0) {
        out_flag[0] = g_volatile_flag;
        out_val[0]  = g_volatile_val;
    }
}

// ------------------------------------------------------------------
// __restrict__ on kernel parameters: hint no aliasing.

__global__ void restrict_add(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// __ldg: load via texture cache (read-only global).

__global__ void ldg_kernel(float *out, const float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]) * 2.0f;
    }
}

// ------------------------------------------------------------------
// __threadfence() and __threadfence_block(): memory fence patterns.

__device__ int g_ready;
__device__ float g_shared_val;

__global__ void producer(float val) {
    if (threadIdx.x == 0) {
        g_shared_val = val;
        __threadfence();
        g_ready = 1;
    }
}

__global__ void consumer(float *out) {
    if (threadIdx.x == 0) {
        while (g_ready == 0) {}  // spin-wait
        out[0] = g_shared_val;
    }
}

// ------------------------------------------------------------------
// clock() intrinsic: read 32-bit timer.

__global__ void timing_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int t0 = clock();
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        int t1 = clock();
        out[0] = sum;
        out[1] = t1 - t0;
    }
}

// ------------------------------------------------------------------
// __threadfence_block(): block-scope fence.

__global__ void fence_block(float *out, float *in, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    if (tid < 32 && tid < n) smem[tid] = in[tid];
    __threadfence_block();
    if (tid < n && tid < 32) {
        out[tid] = smem[(tid + 1) % 32];
    }
}
