// Probe: Preprocessor with stringification and token pasting 
// - #if / #elif / #else / #endif
// - #define with parens and multiple args
// - #undef
// - ## token paste (should be ignored)
// - Macro that expands to a type

#define CUDA_VERSION 12000
#define SM_VERSION 120

#if CUDA_VERSION >= 12000
#define USE_NEW_API 1
#else
#define USE_NEW_API 0
#endif

#if SM_VERSION >= 90
#define HAS_BF16 1
#else
#define HAS_BF16 0
#endif

#define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

__global__ void macro_math(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float vmax = MAX3(a[tid], b[tid], c[tid]);
        float vmin = MIN3(a[tid], b[tid], c[tid]);
        out[tid] = vmax - vmin;
    }
}
