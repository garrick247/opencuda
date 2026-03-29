// Probe: optimizer safety — LICM must not hoist variant computations,
// CSE must not merge computations with different types,
// constant folding edge cases (integer division, negative modulo),
// optimizer interaction with break/continue paths

// LICM safety: computation reads a value that's written in the loop body
// The read-after-write makes it loop-variant — must NOT be hoisted
__global__ void licm_must_not_hoist(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int prev = 0;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int diff = in[i] - prev;   // diff depends on prev (variant)
            sum += diff;
            prev = in[i];              // prev changes each iteration
        }
        *out = sum;
    }
}

// Integer division rounds toward zero (not floor) for negative dividends
__global__ void int_div_truncation(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // In C, -7/2 = -3 (truncates toward zero), not -4 (floor)
        out[tid] = v / 2;
    }
}

// Integer modulo: sign follows dividend in C
__global__ void int_mod_sign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // -7 % 3 = -1 in C (sign follows dividend), not 2 (positive remainder)
        out[tid] = v % 3;
    }
}

// CSE: two loads of in[tid] — should be merged by CSE
__global__ void cse_same_load(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];   // first load
        int b = in[tid];   // second load — same address, should CSE
        out[tid] = a + b;  // should be in[tid] * 2
    }
}

// Constant folding: operations on literals
__global__ void const_arith(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = 3 * 4 + 2;    // should fold to 14
        int y = (100 / 7);     // should fold to 14
        int z = 100 % 13;      // should fold to 9
        out[tid] = x + y + z;  // should be 14 + 14 + 9 = 37
    }
}
