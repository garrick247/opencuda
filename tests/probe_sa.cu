// Probe: device function call conventions — many args, nested calls,
// value passing vs pointer, and device function chains.

// ------------------------------------------------------------------
// Device function with 6 int arguments.

__device__ int sum6(int a, int b, int c, int d, int e, int f) {
    return a + b + c + d + e + f;
}

__global__ void call_sum6(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = sum6(v, v+1, v+2, v+3, v+4, v+5);
    }
}

// ------------------------------------------------------------------
// Device function with mixed float/int arguments.

__device__ float weighted(int idx, float a, float b, float c, float w) {
    return (a + b + c) * w + (float)idx;
}

__global__ void call_weighted(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = weighted(tid, v, v * 2.0f, v * 3.0f, 0.5f);
    }
}

// ------------------------------------------------------------------
// Nested device function calls (chain depth 3).

__device__ int double_val(int x) { return x * 2; }
__device__ int triple_val(int x) { return x * 3; }
__device__ int six_val(int x)    { return double_val(triple_val(x)); }

__global__ void nested_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = six_val(in[tid]) + double_val(in[tid]) + triple_val(in[tid]);
    }
}

// ------------------------------------------------------------------
// Device function returning struct (via out-pointer).

struct MinMax { int lo, hi; };

__device__ MinMax find_minmax(int a, int b, int c) {
    MinMax r;
    r.lo = a < b ? (a < c ? a : c) : (b < c ? b : c);
    r.hi = a > b ? (a > c ? a : c) : (b > c ? b : c);
    return r;
}

__global__ void call_minmax(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid * 3 + 0];
        int b = in[tid * 3 + 1];
        int c = in[tid * 3 + 2];
        MinMax mm = find_minmax(a, b, c);
        out[tid * 2 + 0] = mm.lo;
        out[tid * 2 + 1] = mm.hi;
    }
}

// ------------------------------------------------------------------
// Multiple device function calls in one kernel (register pressure).

__device__ int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__global__ void multi_clamp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = clamp_int(v,       0, 255);
        int b = clamp_int(v - 10, -5,  15);
        int c = clamp_int(v * 2, -100, 100);
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Device function with pointer argument (write-back).

__device__ void swap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

__global__ void call_swap(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid * 2 + 0];
        int y = in[tid * 2 + 1];
        if (x > y) swap(&x, &y);
        out[tid * 2 + 0] = x;
        out[tid * 2 + 1] = y;
    }
}

// ------------------------------------------------------------------
// Device function called in loop (inlining pressure).

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__global__ void lerp_chain(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = v;
        for (int i = 0; i < 4; i++) {
            float t = (float)i * 0.25f;
            r = lerp(r, v * 2.0f, t);
        }
        out[tid] = r;
    }
}
