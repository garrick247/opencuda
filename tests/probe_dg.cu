// Probe: unusual struct patterns that may expose layout issues
// - Struct with array field accessed via computed index in loop
// - Struct with very large array field (>16 elements)
// - Struct in array, passed by value vs pointer

struct Filter {
    float weights[32];
    int size;
    float bias;
};

__device__ float apply_filter_1d(Filter f, float *data, int offset) {
    float sum = f.bias;
    for (int i = 0; i < f.size; i++) {
        sum += f.weights[i] * data[offset + i];
    }
    return sum;
}

__global__ void conv1d(float *out, float *in, float *weights_flat,
                       float bias, int filter_size, int n) {
    int tid = threadIdx.x;
    if (tid + filter_size <= n) {
        Filter f;
        f.size = filter_size;
        f.bias = bias;
        for (int i = 0; i < filter_size && i < 32; i++) {
            f.weights[i] = weights_flat[i];
        }
        out[tid] = apply_filter_1d(f, in, tid);
    }
}
