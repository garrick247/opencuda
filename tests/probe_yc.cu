// Probe: typedef'd function-like types, multi-decl on one line, compound literal
// struct init, sizeof in array sizing, #define with operators, string literal
// (as unreachable dead code), for-loop with no body, while with no body (semicolon),
// chained comparisons (a < b && b < c), integer promotion edge cases,
// unsigned overflow wrapping, and volatile global pointer load.

// ------------------------------------------------------------------
// typedef struct + typedef scalar.

typedef struct { int x, y; } IVec2;
typedef unsigned int uint32;
typedef long long int i64;

__device__ int ivec2_dot(IVec2 a, IVec2 b) {
    return a.x * b.x + a.y * b.y;
}

__global__ void typedef_kernel(int *out, int *ax, int *ay,
                                   int *bx, int *by, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        IVec2 a, b;
        a.x = ax[tid]; a.y = ay[tid];
        b.x = bx[tid]; b.y = by[tid];
        out[tid] = ivec2_dot(a, b);
    }
}

// ------------------------------------------------------------------
// Multi-declaration on one line: int a, b, c;

__global__ void multi_decl(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = in[tid];
        b = a + 1;
        c = b * 2;
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// sizeof in computation.

__global__ void sizeof_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int si = (int)sizeof(int);
        int sf = (int)sizeof(float);
        int sd = (int)sizeof(double);
        int sll = (int)sizeof(long long);
        out[tid] = si + sf + sd + sll;  // 4+4+8+8 = 24
    }
}

// ------------------------------------------------------------------
// #define with expression.

#define SQ(x)     ((x) * (x))
#define CUBE(x)   ((x) * (x) * (x))
#define MAX3(a,b,c) (((a)>(b)) ? (((a)>(c)) ? (a) : (c)) : (((b)>(c)) ? (b) : (c)))

__global__ void macro_expr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = MAX3(SQ(v), CUBE(v), v + 1);
    }
}

// ------------------------------------------------------------------
// for-loop with empty body (semicolon): scan to find first zero.

__global__ void empty_body_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i;
        for (i = 0; i < n && in[i] != 0; i++);
        out[tid] = i;
    }
}

// ------------------------------------------------------------------
// while with empty body (advance pointer to end).

__global__ void empty_while(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int cnt = 0;
        int v = tid;
        // count steps to reduce v to 1 via integer halving
        while (v > 1 && cnt < 32) { v >>= 1; cnt++; }
        out[tid] = cnt;
    }
}

// ------------------------------------------------------------------
// Chained range check: a < b && b < c.

__global__ void chained_cmp(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid], z = c[tid];
        // in range (x, z)?
        int r = (x < y && y < z) ? 1 :
                (x > y && y > z) ? -1 : 0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// unsigned overflow / wrapping arithmetic.

__global__ void unsigned_wrap(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned v = in[tid];
        unsigned wrap = v + 0xFFFFFFFFu;  // wraps to v - 1
        unsigned diff = 0u - v;           // two's complement negation
        out[tid * 2]     = wrap;
        out[tid * 2 + 1] = diff;
    }
}

// ------------------------------------------------------------------
// Integer promotion: char + int, short + int.

__global__ void int_promotion(int *out, signed char *c_in,
                                short *s_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed char cv = c_in[tid];
        short       sv = s_in[tid];
        // Both should promote to int for arithmetic
        int r = (int)cv + (int)sv + 1;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Volatile global load via pointer.

__global__ void volatile_global(int *out, volatile int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = v * 2;
    }
}

// ------------------------------------------------------------------
// typedef enum.

typedef enum { STATE_IDLE = 0, STATE_RUN = 1, STATE_DONE = 2 } State;

__device__ int state_score(State s) {
    switch ((int)s) {
        case STATE_IDLE: return 0;
        case STATE_RUN:  return 10;
        case STATE_DONE: return 100;
        default:         return -1;
    }
}

__global__ void enum_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        State s = (State)(in[tid] % 3);
        out[tid] = state_score(s);
    }
}
