// Probe: __launch_bounds__, __restrict__ qualifier, volatile memory,
// threadIdx in float context, multiple kernels sharing device fn,
// pointer comparison

// __launch_bounds__ — parsed but doesn't affect semantics
__global__ __launch_bounds__(256, 2) void launch_bounded(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = tid;
}

// __restrict__ hint on pointer parameters
__global__ void restricted_ptrs(int * __restrict__ out,
                                  const int * __restrict__ in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] * 3;
}

// Volatile memory access
__global__ void volatile_access(volatile int *out, volatile int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = v + 1;
    }
}

// threadIdx.x used in float computation (implicit cast)
__global__ void tid_in_float(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float ft = (float)threadIdx.x;
        out[tid] = ft * 0.5f;
    }
}

// Multiple kernels sharing the same device function
__device__ int double_it(int x) { return x * 2; }

__global__ void use_double_1(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = double_it(in[tid]);
}

__global__ void use_double_2(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = double_it(in[tid]) + 1;
}

// NULL pointer check pattern (pointer comparison to 0)
__global__ void null_check(int *out, int *opt_in, int n, int default_val) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v;
        if (opt_in != 0) {
            v = opt_in[tid];
        } else {
            v = default_val;
        }
        out[tid] = v;
    }
}
