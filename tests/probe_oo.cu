// Probe: enum values, parameterized macros, typed constants, enum in switch.

// ------------------------------------------------------------------
// Enum values used as array sizes and in comparisons.

enum Color { RED = 0, GREEN = 1, BLUE = 2, NUM_COLORS = 3 };

__global__ void enum_compare(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid] % NUM_COLORS;
        if (v == RED)   out[tid] = 0xFF0000;
        else if (v == GREEN) out[tid] = 0x00FF00;
        else             out[tid] = 0x0000FF;
    }
}

// ------------------------------------------------------------------
// Parameterized #define macros.

#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN2(MAX2((x), (lo)), (hi)))
#define SQUARE(x) ((x) * (x))

__global__ void macro_ops(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        float clamped = CLAMP(v, -1.0f, 1.0f);
        float sq = SQUARE(clamped);
        out[tid] = sq;
    }
}

// ------------------------------------------------------------------
// Enum in switch statement.

enum Op { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2, OP_DIV = 3 };

__global__ void enum_switch(int *out, int *a, int *b, int op, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid];
        int r;
        switch ((Op)op) {
            case OP_ADD: r = x + y; break;
            case OP_SUB: r = x - y; break;
            case OP_MUL: r = x * y; break;
            case OP_DIV: r = (y != 0) ? x / y : 0; break;
            default:     r = 0; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// #define with multi-token replacement (type alias style).

#define FVEC float*
#define IVEC int*
#define IDX(arr, i) ((arr)[(i)])

__global__ void define_alias(FVEC out, IVEC data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (float)IDX(data, tid);
    }
}

// ------------------------------------------------------------------
// Nested macro expansion with side-effect-free argument.

#define ABS(x) ((x) < 0 ? -(x) : (x))
#define MAXABS(a, b) MAX2(ABS(a), ABS(b))

__global__ void nested_macros(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = MAXABS(a[tid], b[tid]);
    }
}
