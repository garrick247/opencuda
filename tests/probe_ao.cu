// Probe: string operations, memcpy/memset patterns,
//        __device__ function pointer table,
//        static local variable (C++: not well-defined for CUDA but should parse),
//        complex initializer expressions

// Static local variable (no-op semantics in GPU, but should parse)
__device__ int get_id(void) {
    static int call_count = 0;  // treated as regular local in GPU
    call_count++;
    return call_count;
}

// Array with complex initializer
__constant__ float c_weights[8] = {
    1.0f / 8.0f,
    2.0f / 8.0f,
    3.0f / 8.0f,
    2.0f / 8.0f,
    1.0f / 8.0f,
    0.5f / 8.0f,
    0.25f / 8.0f,
    0.125f / 8.0f
};

__global__ void weighted_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            int idx = (tid + i) % n;
            sum += c_weights[i] * in[idx];
        }
        out[tid] = sum;
    }
}

// Multiple return type: int2, float2, float4 struct decomposition
struct int2_t {
    int x, y;
};

__device__ int2_t make_int2(int x, int y) {
    int2_t r;
    r.x = x;
    r.y = y;
    return r;
}

__global__ void int2_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int2_t v = make_int2(in[tid], in[tid] * 2);
        out[tid] = v.x + v.y;
    }
}
