// Probe: nested device fn inlining, float-to-int truncation, INT limits,
// compound conditions with function call results.

// ------------------------------------------------------------------
// Nested inlining: f calls g which calls h.
// Tests that 3-level inline correctly chains results.

__device__ int h_fn(int x) { return x * x; }
__device__ int g_fn(int x) { return h_fn(x) + h_fn(x + 1); }
__device__ int f_fn(int a, int b) { return g_fn(a) + g_fn(b); }

__global__ void nested_inline(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = f_fn(data[tid], data[tid] + 1);
    }
}

// ------------------------------------------------------------------
// Float-to-int truncation: (int)f must truncate toward zero (not floor).
// (int)(-2.9f) == -2 (not -3), (int)(2.9f) == 2.

__global__ void float_to_int(int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // cvt.rzi.s32.f32 — round toward zero (truncate)
        out[tid] = (int)data[tid];
    }
}

// ------------------------------------------------------------------
// INT_MAX and INT_MIN constants.
// Tests that these are recognized and emitted as correct values.

__global__ void int_limits(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = 2147483647;   // INT_MAX
        out[1] = -2147483648;  // INT_MIN
        out[2] = 2147483647 + 1;  // overflows to INT_MIN (wraps in s32)
    }
}

// ------------------------------------------------------------------
// Function call result used directly in compound condition.
// if (f(a) && g(b)) — both should be inlined and predicates AND'd.

__device__ int is_positive(int x) { return x > 0; }
__device__ int is_even(int x) { return (x & 1) == 0; }

__global__ void call_in_cond(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        if (is_positive(v) && is_even(v)) {
            out[tid] = v / 2;
        } else {
            out[tid] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Device function that returns double.
// Tests that f64 return types are correctly handled in inline.

__device__ double lerp_d(double a, double b, double t) {
    return a + t * (b - a);
}

__global__ void double_lerp(double *out, double *a, double *b, double t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = lerp_d(a[tid], b[tid], t);
    }
}
