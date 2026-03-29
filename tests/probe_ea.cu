// Probe: Patterns with global scope function-style macros
// - Macro generating multiple statements
// - Macro with __device__ function call inside
// - #define with multiple lines using backslash

#define RELU(x)     ((x) > 0.0f ? (x) : 0.0f)
#define GELU_APPROX(x) ((x) * 0.5f * (1.0f + tanhf(0.7978845608f * ((x) + 0.044715f * (x)*(x)*(x)))))
#define SILU(x)     ((x) / (1.0f + expf(-(x))))

__global__ void activations(float *relu_out, float *silu_out,
                             float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float v = in[tid];
        relu_out[tid] = RELU(v);
        silu_out[tid] = SILU(v);
    }
}

// Multi-step macro that builds an expression
#define LERP(a, b, t) ((a) + ((b) - (a)) * (t))
#define BILERP(a, b, c, d, tx, ty) \
    LERP(LERP(a, b, tx), LERP(c, d, tx), ty)

__global__ void bilerp_kernel(float *out, float a, float b,
                               float c, float d, float *tx, float *ty, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = BILERP(a, b, c, d, tx[tid], ty[tid]);
    }
}
