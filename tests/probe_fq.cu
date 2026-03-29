// Probe: #ifdef / #ifndef / #else / #endif preprocessor conditionals
// Also: #if with numeric comparison, #undef

#define PRECISION 32
#define ENABLE_DOUBLE 0
#define VERSION 3

#ifdef PRECISION
#  define FLOAT_T float
#else
#  define FLOAT_T double
#endif

#ifndef BLOCK_SIZE
#  define BLOCK_SIZE 128
#endif

#if ENABLE_DOUBLE
typedef double compute_t;
#else
typedef float compute_t;
#endif

#if VERSION >= 2
#define USE_FAST_MATH 1
#else
#define USE_FAST_MATH 0
#endif

__global__ void conditional_compile(FLOAT_T *out, FLOAT_T *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        compute_t v = (compute_t)in[tid];
#if USE_FAST_MATH
        v = v * v;
#else
        v = v + 1.0f;
#endif
        out[tid] = (FLOAT_T)v;
    }
}

__global__ void block_size_check(float *out, float *in, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    if (tid < n && tid < BLOCK_SIZE) {
        smem[tid] = in[tid];
        __syncthreads();
        out[tid] = smem[(tid + 1) % BLOCK_SIZE];
    }
}
