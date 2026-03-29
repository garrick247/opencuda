// Probe: nasty edge cases — expressions that should generate
// setp/selp correctly, tricky const promotion, and optimizer-
// sensitive patterns near the per-block boundary.

// ------------------------------------------------------------------
// Integer constant expressions that trigger constant folding.

__global__ void const_fold_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // These should all fold to constants
        int k = (1 + 2) * (3 + 4) - 5;  // = 16
        int m = k * k / 4;               // = 64
        int p = m + k - 32;              // = 48
        out[tid] = v + p;
    }
}

// ------------------------------------------------------------------
// Float constant expressions.

__global__ void float_const_fold(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float k = 2.0f * 3.14159f / 6.0f;  // ≈ pi/3
        float m = k * k + 1.0f;
        out[tid] = v * m;
    }
}

// ------------------------------------------------------------------
// Loop unroll candidate: trip count 8 with carried dependency.

__global__ void unroll_carry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v, b = 1;
        // This loop has two loop-carried vars: a and b
        for (int i = 0; i < 8; i++) {
            int new_a = a + b;
            int new_b = a * 2;
            a = new_a;
            b = new_b;
        }
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// CSE candidate: same expression computed twice in same block.

__global__ void cse_candidate(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid], cv = c[tid];
        float ab = av * bv + cv;       // first use
        float ab2 = av * bv + cv;      // same expr — should be CSE'd
        float cross = av * bv - cv;    // different
        out[tid] = ab + ab2 + cross;  // ab + ab2 = 2*ab
    }
}

// ------------------------------------------------------------------
// Constant propagation through assignments.

__global__ void const_prop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = 5;
        int y = x * 2;      // 10
        int z = y + x;      // 15
        int w = z - x;      // 10
        // v * w should simplify to v * 10
        out[tid] = v * w + z;
    }
}

// ------------------------------------------------------------------
// LICM candidate: loop-invariant expression.

__global__ void licm_candidate(float *out, float *in, float *coeff, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float c0 = coeff[0], c1 = coeff[1], c2 = coeff[2];
        // c0*c1 + c2 is loop-invariant (loaded once before loop)
        float inv = c0 * c1 + c2;
        float acc = 0.0f;
        for (int i = 0; i < 8; i++) {
            acc += v * inv + (float)i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Identity fold opportunity: add/sub 0, mul by 1.

__global__ void identity_ops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v + 0;    // fold: a = v
        int b = v * 1;    // fold: b = v
        int c = v - 0;    // fold: c = v
        int d = 0 + v;    // fold: d = v
        int e = 1 * v;    // fold: e = v
        out[tid] = a + b + c + d + e;  // = 5*v
    }
}

// ------------------------------------------------------------------
// Dead code after return: must not be emitted.

__device__ int early_exit_fn(int v) {
    if (v < 0) return -1;
    if (v == 0) return 0;
    if (v < 10) return 1;
    return 2;
    // Unreachable:
    return 999;  // dead — should not affect result
}

__global__ void dead_code_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = early_exit_fn(in[tid]);
}
