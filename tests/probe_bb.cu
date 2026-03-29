// Probe: edge cases in PTX codegen correctness
// - Large struct passed by value to __device__ function
// - Recursive-looking device functions (non-recursive but same name pattern)
// - Device function with 8+ parameters  
// - Returning struct from device function used directly as expression

struct Matrix4x4 {
    float m[16];
};

__device__ float mat_trace(Matrix4x4 m) {
    return m.m[0] + m.m[5] + m.m[10] + m.m[15];
}

__device__ float sum8(float a, float b, float c, float d,
                      float e, float f, float g, float h) {
    return a + b + c + d + e + f + g + h;
}

__global__ void mat_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 4) {
        Matrix4x4 m;
        for (int i = 0; i < 16; i++) {
            m.m[i] = in[i];
        }
        out[tid] = mat_trace(m);
    }
}

__global__ void sum8_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sum8(in[0], in[1], in[2], in[3],
                        in[4], in[5], in[6], in[7]);
    }
}
