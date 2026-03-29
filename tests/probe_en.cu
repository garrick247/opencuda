// Probe: complex ternary chains, ternary with function calls, nested ternary
// Also: ternary whose branches have different types (int/float promotion)

__device__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ int sign(int x) {
    return x > 0 ? 1 : (x < 0 ? -1 : 0);
}

__global__ void ternary_chain(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Nested ternary
        float r = v > 100.0f ? 100.0f :
                  v < 0.0f   ? 0.0f   :
                  v;
        // Ternary with function call result
        float c = clampf(v, 0.0f, 1.0f);
        out[tid] = r + c;
    }
}

__global__ void sign_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sign(in[tid]);
    }
}

// Ternary assigning to different lhs paths
__global__ void ternary_lhs(float *out, float *a, float *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sel[tid] ? a[tid] : b[tid];
    }
}
