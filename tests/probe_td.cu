// Probe: optimizer edge cases — CSE across consecutive statements,
// LICM candidates that must NOT be hoisted, strength reduction chains,
// and const-fold with negative constants.

// ------------------------------------------------------------------
// CSE: same expression computed twice in same block.

__global__ void cse_double(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * v + v;      // computed once
        int b = v * v + v + 1;  // should reuse v*v+v
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}

// ------------------------------------------------------------------
// CSE: same comparison used in two conditions.

__global__ void cse_cmp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int c1 = (v > 10);
        int c2 = (v > 10) ? v : 0;  // should reuse v>10
        int c3 = (v > 10) & (v < 100);
        out[tid] = c1 + c2 + c3;
    }
}

// ------------------------------------------------------------------
// Strength reduction: x*2 should become x+x or shl.

__global__ void strength_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 2;
        int b = v * 4;
        int c = v * 8;
        int d = v * 16;
        out[tid] = a + b + c + d;
    }
}

// ------------------------------------------------------------------
// Const fold with negative: -1 * x = -x, 0 - x = -x.

__global__ void neg_const_fold(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = -1 * v;
        int b = 0 - v;
        int c = v + (-5);    // const fold: v - 5
        int d = v * (-2);    // neg const
        out[tid] = a + b + c + d;
    }
}

// ------------------------------------------------------------------
// LICM: loop-invariant value that MUST be hoisted correctly.

__global__ void licm_invariant(int *out, int *in, int n, int scale) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int k = scale * scale + 1;  // invariant — scale doesn't change in loop
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            acc += v * k + i;  // v*k is invariant, i is not
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// LICM must NOT hoist: loop-dependent subexpression.

__global__ void licm_dep(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            int x = v + i;       // depends on i — NOT invariant
            acc += x * x;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Identity fold chains: x+0, x*1, x-0, x/1.

__global__ void identity_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v + 0;
        int b = a * 1;
        int c = b - 0;
        int d = c / 1;
        out[tid] = d;
    }
}

// ------------------------------------------------------------------
// Const prop chain: only final result matters.

__global__ void const_prop_chain(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 3;
        int b = a * 7;        // 21
        int c = b + 9;        // 30
        int d = c / 5;        // 6
        int e = d * d - d;    // 30
        out[tid] = e + tid;
    }
}

// ------------------------------------------------------------------
// Dead store: assignment overwritten before use.

__global__ void dead_store(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        r = 999;       // dead
        r = v * 2;     // live
        out[tid] = r;
    }
}
