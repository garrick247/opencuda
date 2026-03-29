// Probe: optimizer edge cases — double identity folds, negative constant
// folding, CSE of computations appearing in both if/else branches,
// LICM safety with conditionally-executed loads, and strength reduction.

// ------------------------------------------------------------------
// Double identity folds: x + 0.0, x * 1.0, x - 0.0, x / 1.0.

__global__ void double_identity(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double a = v + 0.0;   // should fold to v
        double b = v * 1.0;   // should fold to v
        double c = v - 0.0;   // should fold to v
        double d = v / 1.0;   // should fold to v
        out[tid] = a + b + c + d;   // 4v
    }
}

// ------------------------------------------------------------------
// Negative constant folding.

__global__ void neg_const_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = -5 * -3;       // 15
        int b = -8 / -2;       // 4
        int c = -7 + -3;       // -10
        int d = -100 - -50;    // -50
        out[tid] = a + b + c + d;  // 15+4-10-50 = -41
    }
}

// ------------------------------------------------------------------
// Hex constant folding.

__global__ void hex_const_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 0xFF & 0x0F;    // 15
        int b = 0x80 | 0x01;    // 129
        int c = 0xFF ^ 0xAA;    // 85
        out[tid] = a + b + c + tid;   // 229 + tid
    }
}

// ------------------------------------------------------------------
// Strength reduction: multiply by power of 2 → shift.

__global__ void strength_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 2;    // should become shl 1
        int b = v * 4;    // should become shl 2
        int c = v * 8;    // should become shl 3
        int d = v * 16;   // should become shl 4
        out[tid] = a + b + c + d;   // 30v
    }
}

// ------------------------------------------------------------------
// Repeated identical computation in both branches — CSE should fire.

__global__ void cse_both_branches(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int expensive = v * v + v;  // computed once before branch
        int r;
        if (v > 0) {
            r = expensive + 1;
        } else {
            r = expensive - 1;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Loop-invariant but only valid if guard passed: LICM must not hoist
// a load that's only valid inside the condition.

__global__ void licm_guarded_load(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            if (v > 0) {
                int w = in[v & (n - 1)];   // load only valid when v>0 and in-bounds
                sum += w + i;
            } else {
                sum += i;
            }
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Chain of identity folds on float.

__global__ void float_identity_chain(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = v * 1.0f;   // fold
        float b = a + 0.0f;   // fold
        float c = b - 0.0f;   // fold
        float d = c / 1.0f;   // fold
        float e = d * 1.0f;   // fold
        // Should ultimately just be: out[tid] = v
        out[tid] = e;
    }
}

// ------------------------------------------------------------------
// Dead code after unconditional return.

__device__ int early_ret(int v) {
    if (v > 0) return v;
    return 0;
    int dead = 999;   // dead — should never execute
    return dead;
}

__global__ void dead_after_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = early_ret(in[tid]);
    }
}

// ------------------------------------------------------------------
// Associative constant folding across multiple operations.

__global__ void assoc_const_fold(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // All constants should fold together: (1 + 2 + 3 + 4 + 5) = 15
        int r = v + 1 + 2 + 3 + 4 + 5;
        out[tid] = r;
    }
}
