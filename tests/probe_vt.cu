// Probe: #ifdef/#ifndef/#else conditional compilation, nested #define,
// variadic macros, and multi-level cast chains.

// ------------------------------------------------------------------
// #ifdef conditional compilation.

#define USE_FAST_PATH 1

__global__ void ifdef_dispatch(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
#ifdef USE_FAST_PATH
        out[tid] = v * 2.0f;   // fast path: scale
#else
        out[tid] = v + v;      // slow path: add (should be compiled out)
#endif
    }
}

// ------------------------------------------------------------------
// #ifndef and #else.

#undef FEATURE_X
// FEATURE_X is not defined

__global__ void ifndef_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
#ifndef FEATURE_X
        out[tid] = v + 1;    // FEATURE_X not defined, this branch taken
#else
        out[tid] = v - 1;    // should be compiled out
#endif
    }
}

// ------------------------------------------------------------------
// Nested #define: macro that uses another macro.

#define BASE 10
#define SCALE 3
#define OFFSET (BASE * SCALE)

__global__ void nested_define(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] + OFFSET;  // in[tid] + 30
    }
}

// ------------------------------------------------------------------
// Variadic macro (printf-like).

#define LOG_INT(x) printf("val=%d\n", (x))

__global__ void variadic_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (tid == 0) {
            LOG_INT(v);
        }
        out[tid] = v * 2;
    }
}

// ------------------------------------------------------------------
// Multi-level cast chain.

__global__ void cast_chain(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = in[tid];
        // Chain of casts
        int r = (int)(double)(float)f;   // float→float→double→int
        unsigned long long ull = (unsigned long long)(long long)(int)f;
        out[tid] = r + (int)(ull & 0xFF);
    }
}

// ------------------------------------------------------------------
// Macro arithmetic: CLAMP using nested MAX/MIN macros.

#define MY_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MY_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MY_CLAMP(x, lo, hi) (MY_MAX((lo), MY_MIN((x), (hi))))

__global__ void macro_clamp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = MY_CLAMP(v, 0, 255);
    }
}

// ------------------------------------------------------------------
// #define with argument stringification-like pattern.

#define TIMES2(x) ((x) * 2)
#define TIMES4(x) (TIMES2(x) * 2)
#define TIMES8(x) (TIMES4(x) * 2)

__global__ void chained_macro_mul(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = TIMES8(v);  // v * 8
    }
}
