// Probe: variadic macros, __VA_ARGS__, multi-arg macros with side effects

#define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))
#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define SQ(x) ((x)*(x))
#define LERP(a, b, t) ((a) + ((b)-(a))*(t))

__global__ void macro_expr(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = v - 1.0f;
        float b = v;
        float c = v + 1.0f;
        out[tid] = MAX3(a, b, c);
    }
}

__global__ void clamp_kernel(float *out, float *in, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = CLAMP(in[tid], lo, hi);
    }
}

__global__ void sq_lerp(float *out, float *a, float *b, float t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sq_a = SQ(a[tid]);
        float sq_b = SQ(b[tid]);
        out[tid] = LERP(sq_a, sq_b, t);
    }
}
