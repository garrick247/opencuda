// Probe: preprocessor + lexer stress — token pasting ##, stringification #,
// nested macro expansion, variadic-like multi-arg macro, #define with
// parenthesized body, macro used as loop bound, __VA_ARGS__ (if supported),
// hex float constants, very long identifier, comment-in-macro, and
// macro generating struct field access.

// ------------------------------------------------------------------
// Nested macro expansion: inner macro used in outer macro body.

#define OFFSET(base, i) ((base) + (i))
#define FETCH(arr, base, i) ((arr)[OFFSET(base, i)])

__global__ void nested_macro(int *out, int *in, int base, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = FETCH(in, base, tid);
}

// ------------------------------------------------------------------
// Macro used as loop bound.

#define ITER_COUNT 16

__global__ void macro_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < ITER_COUNT; i++) s += in[(tid + i) % n];
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Macro with parenthesized body generating expression.

#define LERP(a, b, t) ((a) + ((b) - (a)) * (t))

__global__ void lerp_kernel(float *out, float *a, float *b, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = LERP(a[tid], b[tid], t[tid]);
}

// ------------------------------------------------------------------
// Hex float constant (0x1.0p0f = 1.0f).

__global__ void hex_float(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Hex float constants (C99 syntax)
        float scale = 0x1.0p+2f;    // = 4.0
        float bias  = 0x1.8p-1f;    // = 0.75
        out[tid] = in[tid] * scale + bias;
    }
}

// ------------------------------------------------------------------
// Very long identifier (stress lexer token buffer).

__global__ void very_long_name_kernel_that_tests_if_the_lexer_handles_long_identifiers_correctly(
    int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] + 1;
}

// ------------------------------------------------------------------
// Multi-level macro generating struct field access.

#define X_OF(v) ((v).x)
#define Y_OF(v) ((v).y)
#define DOT2(a, b) (X_OF(a)*X_OF(b) + Y_OF(a)*Y_OF(b))

struct Vec2f { float x, y; };

__global__ void macro_struct(float *out, float *ax, float *ay,
                               float *bx, float *by, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec2f a, b;
        a.x = ax[tid]; a.y = ay[tid];
        b.x = bx[tid]; b.y = by[tid];
        out[tid] = DOT2(a, b);
    }
}

// ------------------------------------------------------------------
// Negative hex constant and large literal.

__global__ void neg_hex(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mask = 0xFF00FF00;
        int neg  = -0x10;   // = -16
        unsigned big = 0xDEADBEEF;
        out[tid] = (tid & mask) + neg + (int)(big >> 16);
    }
}

// ------------------------------------------------------------------
// #define for array dimension + typedef.

#define BLOCK_SIZE 256
typedef float real_t;

__global__ void typedef_dim(real_t *out, real_t *in, int n) {
    __shared__ real_t smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    if (gid < n) out[gid] = smem[tid] * 2.0f;
}

// ------------------------------------------------------------------
// Comma operator in for-loop update (dual update).

__global__ void comma_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int i, j;
        for (i = 0, j = n-1; i < j; i++, j--) {
            s += in[i % n] + in[j % n];
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Ternary chain used as array index.

__global__ void ternary_index_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Ternary selecting which quarter of array to read
        int base = (v < -100) ? 0 :
                   (v <    0) ? n/4 :
                   (v <  100) ? n/2 :
                                3*n/4;
        out[tid] = in[(base + tid) % n];
    }
}
