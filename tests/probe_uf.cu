// Probe: optimizer correctness at edge cases — const prop through
// branches, LICM with memory ops, CSE with function calls, and
// strength reduction patterns.

// ------------------------------------------------------------------
// Constant folded in both branch arms.

__global__ void const_branch_fold(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 0) {
            // Both arms have foldable expressions
            int a = 3 * 4 + 1;    // should fold to 13
            int b = a * 2 - 6;    // should fold to 20
            r = v + b;
        } else {
            int a = 10 / 2;       // should fold to 5
            int b = a + a;        // should fold to 10
            r = v - b;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Loop with invariant computation that MUST NOT be hoisted
// (depends on loop body side effect via memory).

__global__ void no_hoist_mem(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int i = 0; i < 4; i++) {
            // in[tid*4+i] is memory-dependent — NOT loop-invariant
            float v = in[tid * 4 + i];
            float scale = 2.0f * 3.0f;  // invariant, may fold
            acc += v * scale;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// CSE: same expression computed multiple times in a block.

__global__ void cse_multi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // These should ideally CSE to one computation
        int a = v * 3 + 7;
        int b = v * 3 + 7;
        int c = v * 3 + 7;
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Strength reduction: multiply by power of 2 → shift.
// (Optimizer may or may not do this; test correctness of output.)

__global__ void strength_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 2;    // may reduce to v + v or v << 1
        int b = v * 4;    // may reduce to v << 2
        int c = v * 8;    // may reduce to v << 3
        int d = v * 16;   // may reduce to v << 4
        out[tid] = a + b + c + d;  // = v * 30
    }
}

// ------------------------------------------------------------------
// Dead store elimination: store to local never read.

__global__ void dead_store(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int dead = v * 42;    // computed but never used (dead)
        int live = v + 1;
        (void)dead;           // suppress unused warning — still dead
        out[tid] = live;
    }
}

// ------------------------------------------------------------------
// Identity chain: x + 0, x * 1, x | 0, x & -1 (all-ones).

__global__ void identity_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v + 0;
        int b = a * 1;
        int c = b | 0;
        int d = c & (-1);   // all-ones mask
        int e = d ^ 0;
        int f = e - 0;
        out[tid] = f;       // should equal v
    }
}

// ------------------------------------------------------------------
// Unroll candidate: small fixed-trip loop.

__global__ void unroll_small(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        // Trip count = 4, should unroll
        for (int i = 0; i < 4; i++) {
            acc += v * i;
        }
        out[tid] = acc;  // = v*(0+1+2+3) = 6*v
    }
}
