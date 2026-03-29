// Probe: typedef for scalar types, enum in expressions,
// function-like #define macros, static const in device code

// typedef scalar types
typedef int MyInt;
typedef float MyFloat;
typedef unsigned int uint;

__global__ void typedef_basic(MyInt *out, MyFloat *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        MyInt i = (MyInt)tid;
        MyFloat f = (MyFloat)fin[tid];
        out[tid] = i + (MyInt)f;
    }
}

// typedef uint
__global__ void typedef_uint(uint *out, uint *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint v = in[tid];
        out[tid] = v * 2u;
    }
}

// enum used in device code
typedef enum {
    COLOR_RED   = 0,
    COLOR_GREEN = 1,
    COLOR_BLUE  = 2
} Color;

__global__ void enum_ops(int *out, int *colors, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int c = colors[tid];
        int result;
        if (c == COLOR_RED)   result = 0xFF0000;
        else if (c == COLOR_GREEN) result = 0x00FF00;
        else if (c == COLOR_BLUE)  result = 0x0000FF;
        else result = 0;
        out[tid] = result;
    }
}

// Function-like macro
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, lo, hi) MAX(MIN(x, hi), lo)
#define SQ(x) ((x) * (x))

__global__ void macro_ops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid * 3]     = MAX(v, 0);
        out[tid * 3 + 1] = CLAMP(v, -100, 100);
        out[tid * 3 + 2] = SQ(v);
    }
}

// static const in device code
__device__ static const int WARP_SIZE = 32;
__device__ static const float PI = 3.14159265f;

__global__ void static_const(int *out, float *fout) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = WARP_SIZE;
        fout[0] = PI;
    }
}
