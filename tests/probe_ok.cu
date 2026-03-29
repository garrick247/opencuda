// Probe: LICM correctness, predicate pressure with many conditions,
// multiple nested loops, loop with multiple exits.

// ------------------------------------------------------------------
// LICM: loop-invariant expression should be hoisted.
// scale * stride is loop-invariant — should be computed once, not each iter.

__global__ void licm_check(float *out, float *data, float scale, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            // scale * stride is loop-invariant
            float coeff = scale * (float)stride;
            sum += data[i] * coeff;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Multiple nested loops: outer loop carries inner loop result.
// Tests that loop writeback handles nested carry correctly.

__global__ void nested_loops(int *out, int n, int m) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            int row = 0;
            for (int j = 0; j < m; j++) {
                row += i * m + j;
            }
            total += row;
        }
        out[0] = total;
    }
}

// ------------------------------------------------------------------
// Loop with two early-exit conditions (multiple breaks).
// Tests that both break paths produce correct results.

__global__ void early_exit_two(int *out, int *data, int n, int target) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found = -1;
        for (int i = 0; i < n; i++) {
            if (data[i] == target) { found = i; break; }
            if (data[i] > target * 2) { found = -(i+1); break; }
        }
        out[0] = found;
    }
}

// ------------------------------------------------------------------
// High predicate pressure: many independent comparisons.
// Tests that predicate registers don't alias when there are many conditions.

__global__ void many_predicates(int *out, int a, int b, int c, int d,
                                 int e, int f, int g, int h) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int r = 0;
        if (a > 0) r += 1;
        if (b > 0) r += 2;
        if (c > 0) r += 4;
        if (d > 0) r += 8;
        if (e > 0) r += 16;
        if (f > 0) r += 32;
        if (g > 0) r += 64;
        if (h > 0) r += 128;
        out[0] = r;
    }
}

// ------------------------------------------------------------------
// Loop with float accumulator and integer loop counter.
// Tests that loop writeback doesn't confuse int and float carry vars.

__global__ void mixed_type_loop(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] > 0.0f) {
                sum += data[i];
                count++;
            }
        }
        out[0] = (count > 0) ? sum / (float)count : 0.0f;
    }
}
