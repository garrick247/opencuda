// Probe: #if/#elif/#endif conditionals, multi-level #define chains,
// operator precedence edge cases, and tricky expression parsing.

// ------------------------------------------------------------------
// #if / #elif / #endif preprocessor conditionals.

#define PRECISION 2

__global__ void if_elif(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
#if PRECISION == 1
        out[tid] = v * 1.0f;            // low precision
#elif PRECISION == 2
        out[tid] = v * 1.5f;            // medium precision (compiled)
#elif PRECISION == 3
        out[tid] = v * 2.0f;            // high precision
#else
        out[tid] = v;                   // default
#endif
    }
}

// ------------------------------------------------------------------
// #if defined() and #if !defined().

#define FEATURE_ON 1

__global__ void ifdef_defined(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
#if defined(FEATURE_ON)
        v = v + 10;   // taken
#endif
#if !defined(FEATURE_OFF)
        v = v * 2;    // also taken (FEATURE_OFF not defined)
#endif
        out[tid] = v;  // (in[tid] + 10) * 2
    }
}

// ------------------------------------------------------------------
// Multi-level #define chain: A → B → constant.

#define PI_DIV_180 0.01745329251f
#define DEG2RAD(d) ((d) * PI_DIV_180)
#define RAD2DEG(r) ((r) / PI_DIV_180)

__global__ void deg_rad_convert(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float deg = in[tid];
        float rad = DEG2RAD(deg);
        float back = RAD2DEG(rad);
        out[tid] = back;  // = deg (round-trip)
    }
}

// ------------------------------------------------------------------
// Operator precedence: ternary vs assignment.
// In C: a ? b : c has lower precedence than = on the right,
// but = can appear as RHS of ternary.

__global__ void ternary_prec(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a, b;
        // a = (v > 0) ? v : 0 — standard ternary assignment
        a = (v > 0) ? v : 0;
        // b = (v > 0) ? (v * 2) : (v * 3)
        b = (v > 0) ? v * 2 : v * 3;
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Comma operator as statement (not as argument separator).

__global__ void comma_stmt(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 0, b = 0;
        // Comma in for-update (covered elsewhere), here in expression stmt
        a = 1, b = 2;   // C: evaluates both, result is b; but we just use a,b
        out[tid] = a + b;   // 3
    }
}

// ------------------------------------------------------------------
// Chained comparison operators (NOT transitive in C — evaluates left-to-right).
// 0 < v < 100 in C means (0 < v) < 100, which is always true (0 or 1 < 100).

__global__ void chained_cmp_trap(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // C: (0 < v) produces 0 or 1; then (0 or 1) < 100 is always true
        // This is intentional — testing the parser handles it correctly
        int r = (0 < v) & (v < 100);   // correct way: use & to combine
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Nested macro call chains.

#define ADD_ONE(x) ((x) + 1)
#define DOUBLE(x)  ((x) * 2)
#define SQUARE(x)  ((x) * (x))

__global__ void nested_macros(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // SQUARE(ADD_ONE(DOUBLE(v))) = ((2v+1))^2 = 4v^2 + 4v + 1
        int r = SQUARE(ADD_ONE(DOUBLE(v)));
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// #undef then redefine with different value.

#define SCALE 10
#undef SCALE
#define SCALE 20

__global__ void undef_redefine(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * SCALE;   // should use SCALE=20
    }
}

// ------------------------------------------------------------------
// sizeof in #if (not standard C but test handling).

#define ELEM_SIZE 4    // sizeof(int)

__global__ void sizeof_define(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int stride = ELEM_SIZE;
        out[tid] = in[tid] + stride;
    }
}

// ------------------------------------------------------------------
// Macro stringification pattern (without actual # operator — just functional test).

#define MAX3(a, b, c) (((a) > (b) ? (a) : (b)) > (c) ? ((a) > (b) ? (a) : (b)) : (c))

__global__ void max3_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = in[tid] + 1;
        int c = in[tid] - 1;
        out[tid] = MAX3(a, b, c);   // = in[tid]+1
    }
}
