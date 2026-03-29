// Probe: Preprocessor edge cases
// - #define with multiple tokens (function-like macro with expressions)
// - #define followed by ## token paste (should be ignored or handled)
// - #ifdef / #ifndef / #else / #endif (conditional compilation)
// - Nested #define usage
// - Multiline macro with backslash continuation

#define SQUARE(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
#define THREADS 128

#ifndef MY_EPSILON
#define MY_EPSILON 1e-7f
#endif

__global__ void poly_eval(float *out, float *x, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = x[tid];
        out[tid] = CUBE(v) + SQUARE(v) + v + MY_EPSILON;
    }
}

#define DOT2(ax, ay, bx, by) ((ax)*(bx) + (ay)*(by))

__global__ void dot2_kernel(float *out, float *ax, float *ay, float *bx, float *by, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = DOT2(ax[tid], ay[tid], bx[tid], by[tid]);
    }
}
