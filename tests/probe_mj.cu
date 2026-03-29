// Probe: Multiple inlined device calls in same statement / expression
// - Two device calls chained in arithmetic: f(x) + g(x)
// - Device call result used in another device call: f(g(x))
// - Device call result stored to struct field
// - Two device calls on same line to different output fields
// - Loop with two inlined calls per iteration

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + x * x);  // approx, avoid expf
}

__device__ float scale_bias(float x, float w, float b) {
    return x * w + b;
}

// f(x) + g(x) in one expression
__global__ void two_calls_add(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = relu(in[tid]) + sigmoid(in[tid]);
    }
}

// f(g(x)) chained
__global__ void chained_calls(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = relu(sigmoid(in[tid]));
    }
}

// Device call result in arithmetic with another device call
__global__ void mixed_calls(float *out, float *in, float w, float b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = relu(scale_bias(x, w, b)) * sigmoid(x);
    }
}

// Two calls in same loop iteration
__global__ void loop_two_calls(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += relu(in[i]) + sigmoid(in[i]);
        }
        out[0] = sum;
    }
}

// Call result assigned to struct fields
struct Activation {
    float r;
    float s;
};

__device__ Activation activate(float x) {
    Activation a;
    a.r = relu(x);
    a.s = sigmoid(x);
    return a;
}

__global__ void struct_from_two_calls(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Activation a = activate(in[tid]);
        out[tid] = a.r + a.s;
    }
}
