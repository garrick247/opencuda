// Probe: optimizer boundary conditions — CSE same-block repeat,
// strength reduction via shifts, identity fold chains, and
// loop with multiple loop-invariant computations.

// ------------------------------------------------------------------
// Same expression computed multiple times in one block — should CSE.

__global__ void cse_repeat(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = a[tid];
        float y = b[tid];
        // Repeated computation in same block
        float sum1 = x + y;
        float prod = x * y;
        float sum2 = x + y;  // same as sum1 — should CSE
        float prod2 = x * y; // same as prod — should CSE
        out[tid] = sum1 * prod + sum2 + prod2;
    }
}

// ------------------------------------------------------------------
// Identity fold chain: x+0, x*1, x-0, x/1.

__global__ void identity_folds(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v + 0;
        int b = v * 1;
        int c = v - 0;
        int d = v / 1;
        // Also: 0+v, 1*v
        int e = 0 + v;
        int f = 1 * v;
        out[tid] = a + b + c + d + e + f;
    }
}

// ------------------------------------------------------------------
// Strength reduction: multiply by power-of-2 → shl.

__global__ void mul_pow2(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 2;    // → shl 1
        int b = v * 4;    // → shl 2
        int c = v * 8;    // → shl 3
        int d = v * 16;   // → shl 4
        int e = v * 32;   // → shl 5
        out[tid] = a + b + c + d + e;
    }
}

// ------------------------------------------------------------------
// Loop with multiple loop-invariant values — all should be hoisted.

__global__ void licm_multi_inv(float *out, float *in, float *coeffs, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // These are all loop-invariant (no loop-carried dependency)
        float c0 = coeffs[0];
        float c1 = coeffs[1];
        float c2 = coeffs[2];
        float scaled = v * c0;        // loop-invariant
        float bias   = c1 + c2;       // loop-invariant
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            // scaled and bias are loop-invariant — used each iter
            acc += scaled * (float)i + bias;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Constant folding of chained constant arithmetic.

__global__ void const_chain_fold(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // All constants — should fold to single value 42
        int c = 2 + 3;           // 5
        int d = c * 4;           // 20
        int e = d + 10;          // 30
        int f = e + 12;          // 42
        out[tid] = v + f;
    }
}

// ------------------------------------------------------------------
// Dead code with side-effect-free computation.

__global__ void dead_pure_code(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int dead1 = v * 3 + 7;   // computed, never used
        int dead2 = v ^ 0xDEAD;  // computed, never used
        (void)dead1;
        (void)dead2;
        out[tid] = v + 1;        // only this matters
    }
}

// ------------------------------------------------------------------
// Boolean expression simplification: double negation, x && true, etc.

__global__ void bool_simplify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = !(!( v > 0));    // double negation — should simplify to (v > 0)
        int b = (v > 0) ? 1 : 0; // explicit 0/1 from predicate
        int c = (v != 0) ? v : 0; // v if nonzero, else 0
        out[tid] = a + b + c;
    }
}
