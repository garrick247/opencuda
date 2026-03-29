// Probe: preprocessor edge cases — multi-level macros, function-like
// macros with multiple arguments, stringize, and conditional macros.

#define PI 3.14159265358979f
#define TWO_PI (2.0f * PI)
#define HALF_PI (PI * 0.5f)
#define SQ(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(v, lo, hi) (MAX2(MIN2((v), (hi)), (lo)))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define SWAP(a, b) do { int _t = (a); (a) = (b); (b) = _t; } while (0)
#define UNROLL4(body) body(0) body(1) body(2) body(3)

// ------------------------------------------------------------------
// Math with multi-level macro expansion.

__global__ void macro_math(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = SQ(v) + CUBE(v) + SQ(PI);
        r = CLAMP(r, 0.0f, 1000.0f);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Macro used as loop body generator (X-macro style).

#define ACCUM_STEP(i) acc += in[tid * 8 + (i)];

__global__ void macro_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        UNROLL4(ACCUM_STEP)  // Expands to 4 separate statements
        acc += in[tid * 8 + 4];
        acc += in[tid * 8 + 5];
        acc += in[tid * 8 + 6];
        acc += in[tid * 8 + 7];
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Conditional compilation.

#define USE_FAST_APPROX 1

__global__ void conditional_compile(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r;
#if USE_FAST_APPROX
        r = __expf(v);
#else
        r = expf(v);
#endif
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Macro with do-while(0) — common idiom for multi-statement macros.

__global__ void swap_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid + 1 < n) {
        int a = in[tid];
        int b = in[tid + 1];
        SWAP(a, b);
        out[tid]     = a;
        out[tid + 1] = b;
    }
}

// ------------------------------------------------------------------
// Multi-arg macro in complex expression.

__global__ void multi_arg_macro(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = MAX2(v, -v);                // abs(v)
        float b = MIN2(SQ(v), 100.0f);        // min(v^2, 100)
        float c = CLAMP(v * TWO_PI, -PI, PI); // angle clamping
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Constant macro expression — should constant-fold.

#define TILE_W 16
#define TILE_H 16
#define TILE_SIZE (TILE_W * TILE_H)

__global__ void tiled_index(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int tile_idx = tid / TILE_SIZE;
        int local_idx = tid % TILE_SIZE;
        int row = local_idx / TILE_W;
        int col = local_idx % TILE_W;
        out[tid] = tile_idx * 1000 + row * 100 + col;
    }
}
