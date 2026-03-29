// Probe: loop unroller edge cases — trip counts at/near 16,
// unrolled loops with CSE, constant-folded trip counts, and
// loops with multiple loop-carried variables after unrolling.

// ------------------------------------------------------------------
// Trip count = 4: fully unrollable, should produce 4 inlined iterations.

__global__ void unroll4(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Trip count = 16: at the limit, should still unroll.

__global__ void unroll16(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 16; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Trip count = 17: just above limit, must NOT unroll.
// Should emit a normal loop.

__global__ void no_unroll17(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 17; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Unrolled loop with multiple loop-carried vars.
// Both a and b must be chained correctly across unrolled iterations.

__global__ void unroll_multivar(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 1;
        for (int i = 0; i < 8; i++) {
            int v = data[i];
            a += v;
            b ^= v;
        }
        out[0] = a + b;
    }
}

// ------------------------------------------------------------------
// Unrolled loop with CSE opportunity: same address computed each iter.
// data[i*2] and data[i*2 + 1] — if unrolled, the CSE in constant-folded
// trip counts should fold i*2 to a constant.

__global__ void unroll_cse(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += data[i * 2] + data[i * 2 + 1];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Unrolled loop with conditional inside.
// Tests that per-iteration conditionals work after unrolling.

__global__ void unroll_cond(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos = 0, neg = 0;
        for (int i = 0; i < 8; i++) {
            int v = data[i];
            if (v >= 0) pos += v;
            else        neg += v;
        }
        out[0] = pos - neg;
    }
}

// ------------------------------------------------------------------
// Nested loops where inner is unrollable, outer is not.
// Inner trip = 4 (unrollable), outer trip = n (runtime).

__global__ void partial_unroll(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            // Inner loop: trip count 4, unrollable
            int row_sum = 0;
            for (int j = 0; j < 4; j++) {
                row_sum += data[i * 4 + j];
            }
            total += row_sum;
        }
        out[0] = total;
    }
}

// ------------------------------------------------------------------
// Loop with step > 1: trip count calculable?
// for (int i = 0; i < 8; i += 2) — 4 iters, should unroll.

__global__ void unroll_step2(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 8; i += 2) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Loop with variable used after unrolled region.
// The loop-exit value of `sum` must be correct after unrolling.

__global__ void unroll_exit_value(int *out, int *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 6; i++) {
            sum += data[i] * (i + 1);
        }
        // sum used after loop
        out[0] = sum * 2;
    }
}
