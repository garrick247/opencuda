// Probe: float comparisons (setp.lt/gt/eq.f32),
// three-way && chain (short-circuit over three conditions),
// comparison result used as integer (bool-to-int),
// struct with float members,
// float ternary with mismatched int/float arms

// Float comparisons: clamp to [lo, hi]
__global__ void float_clamp(float *out, float *in, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[tid] = v;
    }
}

// Float equality and inequality
__global__ void float_eq_check(int *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Counts: exact equal, a < b, a > b
        out[tid * 3 + 0] = (a[tid] == b[tid]) ? 1 : 0;
        out[tid * 3 + 1] = (a[tid] <  b[tid]) ? 1 : 0;
        out[tid * 3 + 2] = (a[tid] >  b[tid]) ? 1 : 0;
    }
}

// Three-way && chain: all three conditions must be true
__global__ void triple_and(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // True only if all three positive
        int all_pos = (a[tid] > 0) && (b[tid] > 0) && (c[tid] > 0);
        out[tid] = all_pos;
    }
}

// Comparison result used as int: count how many are positive
__global__ void count_positive(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            count += (in[i] > 0);   // bool-to-int: 1 if positive, 0 otherwise
        }
        *out = count;
    }
}

// Struct with float members
struct Vec2f {
    float x;
    float y;
};

__global__ void vec2f_dot(float *out, struct Vec2f *a, struct Vec2f *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec2f av = a[tid];
        struct Vec2f bv = b[tid];
        out[tid] = av.x * bv.x + av.y * bv.y;
    }
}
