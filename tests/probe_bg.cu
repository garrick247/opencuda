// Probe: __device__ function called from inside loops, conditionals,
//        and as argument to another __device__ function call

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ float tanh_approx(float x) {
    // Fast tanh: 2*sigmoid(2x) - 1
    return 2.0f * sigmoid(2.0f * x) - 1.0f;
}

__global__ void activation_chain(float *out, float *in, int n, int mode) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r;
        if (mode == 0) {
            r = relu(v);
        } else if (mode == 1) {
            r = sigmoid(v);
        } else {
            r = tanh_approx(v);
        }
        out[tid] = r;
    }
}

// Device function as argument to another device function
__device__ float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

__global__ void nested_func_call(float *out, float *a, float *b, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sigmoid(lerp(a, b, t)) 
        out[tid] = sigmoid(lerp(a[tid], b[tid], sigmoid(t[tid])));
    }
}
