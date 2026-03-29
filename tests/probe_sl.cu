// Probe: inline device functions, static qualifiers, complex preprocessor
// macros that expand to statements, and function-like macros.

// ------------------------------------------------------------------
// Inline device function (explicit __forceinline__).

__device__ __forceinline__ int fast_abs(int x) {
    return x < 0 ? -x : x;
}

__device__ __forceinline__ float fast_clamp(float v, float lo, float hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

__global__ void inline_ops(int *iout, float *fout, int *iin, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        iout[tid] = fast_abs(iin[tid]);
        fout[tid] = fast_clamp(fin[tid], 0.0f, 1.0f);
    }
}

// ------------------------------------------------------------------
// Static const inside kernel (treated as constant).

__global__ void static_const(int *out, int *in, int n) {
    const int SHIFT = 3;
    const float SCALE = 0.125f;
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = (v >> SHIFT) + (int)(v * SCALE);
    }
}

// ------------------------------------------------------------------
// Function-like macro expanding to compound expression.

#define RELU(x)        ((x) > 0 ? (x) : 0)
#define SIGMOID_APPROX(x) (0.5f + (x) * 0.25f)  // Linear approximation
#define LEAKY_RELU(x)  ((x) > 0 ? (x) : (x) * 0.01f)

__global__ void activation_fns(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = RELU(v) + SIGMOID_APPROX(v) + LEAKY_RELU(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Multi-line macro (function-like, expands to expression).

#define LERP(a, b, t)  ((a) + ((b) - (a)) * (t))
#define BILERP(a,b,c,d,u,v) LERP(LERP(a,b,u), LERP(c,d,u), v)

__global__ void bilinear(float *out, float *corners, float *u, float *v, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = corners[tid * 4 + 0];
        float b = corners[tid * 4 + 1];
        float c = corners[tid * 4 + 2];
        float d = corners[tid * 4 + 3];
        out[tid] = BILERP(a, b, c, d, u[tid], v[tid]);
    }
}

// ------------------------------------------------------------------
// Device function called via inline wrapper.

__device__ __forceinline__ int popcount32(unsigned int v) {
    return __popc(v);
}

__global__ void bitcount_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = popcount32(in[tid]);
    }
}

// ------------------------------------------------------------------
// Nested inline calls.

__device__ __forceinline__ float sq(float x) { return x * x; }
__device__ __forceinline__ float dist2(float x, float y) { return sq(x) + sq(y); }
__device__ __forceinline__ float dist3(float x, float y, float z) {
    return dist2(x, y) + sq(z);
}

__global__ void distance_kernel(float *out, float *x, float *y, float *z, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dist3(x[tid], y[tid], z[tid]);
    }
}

// ------------------------------------------------------------------
// __noinline__ device function (hint to not inline).

__device__ __noinline__ int heavy_compute(int v) {
    int r = v;
    for (int i = 0; i < 8; i++) r = r * 3 + i;
    return r;
}

__global__ void noinline_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = heavy_compute(in[tid]);
}
