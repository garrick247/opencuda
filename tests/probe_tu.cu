// Probe: forward-declared device functions (defined after use),
// recursive-style patterns (no actual recursion, just deep call chains),
// and complex expression statements.

// Forward declarations (defined later in file)
__device__ int fwd_a(int x);
__device__ int fwd_b(int x);
__device__ float fwd_c(float x, int n);

// ------------------------------------------------------------------
// Kernel using forward-declared device functions.

__global__ void fwd_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = fwd_a(fwd_b(v));
    }
}

__global__ void fwd_float_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fwd_c(in[tid], tid % 4 + 1);
    }
}

// Definitions after use
__device__ int fwd_b(int x) {
    return x * x + 1;
}

__device__ int fwd_a(int x) {
    return (x > 100) ? 100 : x;
}

__device__ float fwd_c(float x, int n) {
    float r = x;
    for (int i = 0; i < n; i++) {
        r = r * 0.9f + 0.1f;
    }
    return r;
}

// ------------------------------------------------------------------
// Expression statement with side effects only.

__global__ void expr_stmt(int *out, int *counter, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Expression statement: result discarded, only side effect matters
        atomicAdd(counter, (v > 0) ? 1 : 0);
        out[tid] = v * 2;
    }
}

// ------------------------------------------------------------------
// Nested ternary in complex expression.

__global__ void deep_ternary(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        // 3-way ternary: max3
        int m = (av > bv) ? ((av > cv) ? av : cv) : ((bv > cv) ? bv : cv);
        out[tid] = m;
    }
}

// ------------------------------------------------------------------
// Multiple assignments in one statement (via comma — supported?).

__global__ void multi_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = 0, c = 0;
        a = v;
        b = v * 2;
        c = a + b;
        out[tid] = c;
    }
}

// ------------------------------------------------------------------
// Complex initializer: multiple variables with dependencies.

__global__ void dep_init(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = v * v;
        float b = a + v;
        float c = b * a - v;
        float d = (a + b + c) / 3.0f;
        out[tid] = d;
    }
}

// ------------------------------------------------------------------
// Function call result used as lvalue indirection.

__device__ int *pick_array(int *a, int *b, int cond) {
    return cond ? a : b;
}

__global__ void pick_and_write(int *a, int *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *chosen = pick_array(a, b, sel[tid] > 0);
        chosen[tid] = tid * 42;
    }
}
