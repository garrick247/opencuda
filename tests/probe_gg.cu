// Probe: complex expression tree with many operands — stress test for
// register allocation, constant folding, and CSE

__global__ void deep_expr(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid], cv = c[tid];
        // Deep expression tree
        float r = ((av + bv) * (bv - cv) + (av * cv - bv * bv)) /
                  ((av * av + bv * bv + cv * cv) + 1.0f) *
                  ((av - bv) * (bv - cv) * (av - cv));
        out[tid] = r;
    }
}

// Many unique subexpressions (tests register pressure)
__global__ void register_pressure(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float v1 = v + 1.0f;
        float v2 = v * 2.0f;
        float v3 = v - 1.0f;
        float v4 = v / 2.0f;
        float v5 = v1 + v2;
        float v6 = v3 - v4;
        float v7 = v5 * v6;
        float v8 = v1 * v3;
        float v9 = v2 * v4;
        float v10 = v8 + v9;
        float v11 = v7 - v10;
        float v12 = v11 * v5;
        float v13 = v12 / (v6 + 1.0f);
        out[tid] = v13;
    }
}

// Common subexpressions (tests CSE)
__global__ void cse_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Same subexpressions used multiple times
        float sq = v * v;
        float cb = sq * v;
        float r = sq + cb + sq * sq + cb * v;
        out[tid] = r;
    }
}
