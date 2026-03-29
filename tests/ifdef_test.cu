// Regression: #ifdef / #ifndef / #else / #endif / #undef conditional compilation.
// Without fix: both branches of #ifdef were included (only directives stripped,
// not the conditional bodies), producing duplicate or dead code.
#define ENABLE_FAST_PATH 1
#define TILE_SIZE 16

#ifdef ENABLE_FAST_PATH
#define COMPUTE(a, b) ((a) * (b) + (a))
#else
#define COMPUTE(a, b) ((a) + (b))
#endif

__global__ void ifdef_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
#ifdef ENABLE_FAST_PATH
        float r = COMPUTE(v, 2.0f);   // fast path: v*2 + v = 3v
#else
        float r = v + 1.0f;           // slow path (excluded)
#endif
#ifndef NDEBUG
        // Debug mode code (NDEBUG not defined, so this IS included)
        r = r + 0.0f;  // no-op, just to test #ifndef
#endif
        out[tid] = r;
    }
}

#undef ENABLE_FAST_PATH
// After undef: ENABLE_FAST_PATH is no longer defined.
// This kernel should use the fallback define.
#ifndef ENABLE_FAST_PATH
#define FALLBACK_OP(x) ((x) * 2.0f)
#endif

__global__ void undef_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = FALLBACK_OP(in[tid]);
    }
}
