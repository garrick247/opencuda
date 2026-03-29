// Probe: sizeof operator, enum types, __host__ declaration skipping,
// static inline helper functions, and function-like macro with args.

// ------------------------------------------------------------------
// sizeof operator in expressions.

__global__ void sizeof_expr(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sizeof should fold to constants
        int si = sizeof(int);           // 4
        int sf = sizeof(float);         // 4
        int sd = sizeof(double);        // 8
        int sll = sizeof(long long);    // 8
        out[tid] = si + sf + sd + sll;  // 24
    }
}

// ------------------------------------------------------------------
// sizeof of struct type.

struct Pair { int a; int b; };

__global__ void sizeof_struct(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sp = sizeof(struct Pair);   // 8
        out[tid] = sp;
    }
}

// ------------------------------------------------------------------
// Enum type: define and use enum values.

enum Direction { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };

__global__ void enum_use(int *out, int *dirs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int d = dirs[tid];
        int r;
        if (d == NORTH)      r = 0;
        else if (d == SOUTH) r = 1;
        else if (d == EAST)  r = 2;
        else if (d == WEST)  r = 3;
        else                 r = -1;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Enum with explicit values.

enum Status {
    STATUS_OK    = 200,
    STATUS_ERROR = 400,
    STATUS_NULL  = 0
};

__global__ void enum_explicit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v == STATUS_OK)    r = 1;
        else if (v == STATUS_ERROR) r = -1;
        else if (v == STATUS_NULL)  r = 0;
        else r = 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// __host__ function — parser should skip it entirely.

__host__ void cpu_helper(int *p, int n) {
    for (int i = 0; i < n; i++) p[i] = i;
}

__global__ void kernel_after_host(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] + 1;
    }
}

// ------------------------------------------------------------------
// Static inline helper — should be inlinable.

static inline int clamp_val(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void static_inline_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = clamp_val(in[tid], 0, 255);
    }
}

// ------------------------------------------------------------------
// Function-like macro that expands to a compound expression.

#define LERP(a, b, t)  ((a) + ((b) - (a)) * (t))
#define CLAMP01(x)     ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))

__global__ void macro_call(float *out, float *a, float *b, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float ta = a[tid], tb = b[tid], tc = CLAMP01(t[tid]);
        out[tid] = LERP(ta, tb, tc);
    }
}
