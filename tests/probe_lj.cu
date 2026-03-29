// Probe: device fn with guard return (single if return + fallthrough code),
// device fn calling device fn (nested inline),
// __device__ global variable,
// device fn with multiple parameters shadowing outer names

// Device function with guard-style early return (known limitation area)
// if (cond) return x;  ← mid-function return
// ... more code ...
// return y;
__device__ int guard_return(int x, int limit) {
    if (x > limit) return limit;   // early return
    if (x < 0) return 0;           // second early return
    return x;                      // normal return
}

__global__ void call_guard_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = guard_return(in[tid], 100);
    }
}

// Nested device function call: outer calls inner
__device__ int inner_fn(int x) {
    return x * x + 1;
}

__device__ int outer_fn(int x, int y) {
    return inner_fn(x) + inner_fn(y);
}

__global__ void nested_device_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = outer_fn(in[tid], tid);
    }
}

// __device__ global variable
__device__ int g_counter = 42;

__global__ void read_device_global(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = g_counter;
    }
}

// Device fn parameter names shadowing outer kernel names
__device__ int shadow_fn(int n, int tid) {
    // 'n' and 'tid' here are the fn params, not outer kernel's vars
    return n + tid;
}

__global__ void shadow_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Call shadow_fn with args in different order
        out[tid] = shadow_fn(tid, n);   // = tid + n
    }
}
