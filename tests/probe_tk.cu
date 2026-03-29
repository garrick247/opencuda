// Probe: implicit float/double mixing, PHI node correctness with partial
// assignments, loop-carried variable partially updated before break.

// ------------------------------------------------------------------
// Float/double implicit promotion in arithmetic.

__global__ void float_double_mix(double *out, float *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // float + double: float should be promoted to double
        double r = a[tid] + b[tid];
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float result from double computation.

__global__ void double_to_float_result(float *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double computed = v * v + v * 2.0 + 1.0;  // all double
        out[tid] = (float)computed;  // explicit narrow
    }
}

// ------------------------------------------------------------------
// PHI correctness: var assigned only in then-branch, default needed.

__global__ void phi_partial(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int bonus = 0;  // default value
        if (v > 100) {
            bonus = v - 100;  // only assigned in one branch
        }
        // else: bonus stays 0
        out[tid] = v + bonus;  // phi needed
    }
}

// ------------------------------------------------------------------
// PHI with 3 paths: assigned in 2 of 3.

__global__ void phi_two_of_three(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int extra = 0;
        if (v > 200) {
            extra = 100;
        } else if (v > 100) {
            extra = 50;
        }
        // else: extra stays 0
        out[tid] = v + extra;
    }
}

// ------------------------------------------------------------------
// Loop-carried var partially updated before break.

__global__ void partial_loop_carry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int best = -1;
        int best_i = -1;
        for (int i = 0; i < 16; i++) {
            int x = v * (i + 1) - 50;
            if (x < 0) continue;
            if (x > 200) break;  // break before both vars are updated in this iter
            best = x;
            best_i = i;
        }
        // best and best_i should hold values from last complete iteration
        out[tid * 2 + 0] = best;
        out[tid * 2 + 1] = best_i;
    }
}

// ------------------------------------------------------------------
// Do-while with multiple exit conditions.

__global__ void do_while_multi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        int acc = v;
        do {
            acc = (acc * 3 + 1) % 100;
            count++;
        } while (acc != v && count < 50);
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Loop where same variable is updated in all paths (no phi needed).

__global__ void no_phi_needed(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        for (int i = 0; i < 4; i++) {
            // r is always assigned before use in this iteration
            if (v + i > 10) {
                r = v + i - 10;
            } else {
                r = -(v + i);
            }
            v = r;  // v updated every iteration
        }
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Nested loops: inner loop modifies outer loop variable (loop fusion pattern).

__global__ void nested_var_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int total = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                total += v + i * 4 + j;
            }
        }
        out[tid] = total;
    }
}
