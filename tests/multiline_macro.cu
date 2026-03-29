// Regression: multi-line macros with backslash continuation
// Without fix: preprocessor split on '\n' without joining '\'-continued lines →
//   multi-line macro body truncated to '\', remaining lines treated as code →
//   ParseError "unexpected token ','".
// Fix: preprocess() joins lines ending with '\' before directive processing
//   (standard C line-splicing per C99 §5.1.1.2 translation phase 2).

#define COMPLEX_MACRO(a, b, c) \
    ((a) * (a) + \
     (b) * (b) + \
     (c) * (c))

#define LOAD_AND_SCALE(ptr, idx, scale) \
    ((ptr)[(idx)] * (scale))

#define MAX3(x, y, z) \
    ((x) > (y) ? \
        ((x) > (z) ? (x) : (z)) : \
        ((y) > (z) ? (y) : (z)))

__global__ void multiline_macro_test(float *out, float *x, float *y, float *z, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float scale = 0.5f;
        float dot = COMPLEX_MACRO(
            LOAD_AND_SCALE(x, tid, scale),
            LOAD_AND_SCALE(y, tid, scale),
            LOAD_AND_SCALE(z, tid, scale)
        );
        out[tid] = dot;
    }
}

__global__ void max3_test(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = MAX3(a[tid], b[tid], c[tid]);
    }
}
