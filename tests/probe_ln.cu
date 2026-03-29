// Probe: LICM correctness — expressions that look invariant but aren't,
// CSE safety with loop-carried accumulator,
// loop with both LICM candidate and loop-carried variable,
// back-to-back loops sharing a common subexpression

// LICM candidate: scale*scale is loop-invariant, but in[i] is not
// The sum uses scale*scale as a multiplier — should be correct after hoisting
__global__ void licm_correct(int *out, int *in, int n, int scale) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int inv = scale * scale;   // loop-invariant
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i] * inv;    // in[i] is NOT invariant
        }
        out[0] = sum;
    }
}

// NOT invariant: expression depends on loop variable i
__global__ void no_licm(int *out, int *in, int n, int base) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int x = base + i;      // NOT invariant — depends on i
            sum += in[x];
        }
        out[0] = sum;
    }
}

// Loop with BOTH LICM candidate AND loop-carried variable
// factor = a * b (invariant), sum is loop-carried
__global__ void licm_plus_carry(int *out, int *in, int n, int a, int b) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int factor = a * b;   // invariant — LICM should hoist
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i] * factor;
        }
        out[0] = sum;
    }
}

// Two sequential loops sharing a subexpression
// The second loop doesn't "see" the first loop's results
__global__ void two_seq_loops(int *out, int *in, int n, int mult) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Loop 1: sum of in[i] * mult
        int sum1 = 0;
        for (int i = 0; i < n; i++) {
            sum1 += in[i] * mult;
        }
        // Loop 2: same pattern, separate accumulator
        int sum2 = 0;
        for (int i = 0; i < n; i++) {
            sum2 += in[i] * mult;   // same expression
        }
        out[0] = sum1;
        out[1] = sum2;
    }
}

// CSE + loop: verify the accumulator isn't CSEd away
__global__ void cse_loop_safety(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            sum = sum + v;   // sum changes each iter — must NOT be CSEd to same val
        }
        out[0] = sum;
    }
}
