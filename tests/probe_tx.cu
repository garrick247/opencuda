// Probe: loop unroller edge cases — trip count at boundary (16),
// loop with zero trip count, loop with trip count > 16 (no unroll),
// and loop with constant-folded body.

// ------------------------------------------------------------------
// Loop with exactly 16 iterations (at the unroll limit).

__global__ void loop_16(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 16; i++) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop with 17 iterations (one over limit — no unroll).

__global__ void loop_17(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 17; i++) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop with 1 iteration.

__global__ void loop_1(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 1; i++) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop where body can be completely constant-folded.

__global__ void loop_const_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            acc += i * i;  // 0 + 1 + 4 + 9 = 14 — fully constant
        }
        out[tid] = acc + tid;  // 14 + tid
    }
}

// ------------------------------------------------------------------
// Nested loops: inner loop always trips the same count.

__global__ void nested_const_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                acc += v + i + j;
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop where some iterations are dead code.

__global__ void loop_dead_iter(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            if (i < 0) {
                acc += 9999;  // dead — i is never < 0 in this loop
            } else {
                acc += i;
            }
        }
        out[tid] = acc;  // should be 0+1+2+3+4+5+6+7 = 28
    }
}

// ------------------------------------------------------------------
// Loop variable used outside loop (after the loop exits normally).

__global__ void loop_exit_value(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i;
        for (i = 0; i < 8; i++) {
            if (v + i > 100) break;
        }
        // i is the index where we broke, or 8 if no break
        out[tid] = i;
    }
}

// ------------------------------------------------------------------
// Loop with multiple exit variables.

__global__ void multi_exit_var(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0, count = 0;
        for (int i = 0; i < 8; i++) {
            int x = v + i;
            if (x > 50) break;
            sum   += x;
            count += 1;
        }
        out[tid * 2 + 0] = sum;
        out[tid * 2 + 1] = count;
    }
}
