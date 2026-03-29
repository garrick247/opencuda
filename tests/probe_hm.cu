// Probe: do-nothing edge cases that should parse cleanly —
// empty function body, early return in middle of kernel,
// nested function calls (f(g(x))), function call as condition

__device__ __forceinline__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ __forceinline__ float sigmoid_approx(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Nested function calls: sigmoid(relu(x))
__global__ void nested_calls(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sigmoid_approx(relu(in[tid]));
    }
}

// Function call as array index
__device__ int clamp_idx(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

__global__ void safe_gather(float *out, float *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[clamp_idx(idx[tid], n)];
    }
}

// Function call in if condition
__device__ bool is_even(int x) {
    return (x & 1) == 0;
}

__global__ void conditional_fn(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (is_even(in[tid])) {
            out[tid] = in[tid] / 2;
        } else {
            out[tid] = in[tid] * 3 + 1;
        }
    }
}
