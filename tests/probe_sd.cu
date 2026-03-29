// Probe: phi-node stress — values live across multiple control flow paths,
// loop exit values used after the loop, and interleaved predicate/value flow.

// ------------------------------------------------------------------
// Loop exit value: last assigned value from early-exit loop.

__global__ void loop_exit_val(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int found = -1;
        int found_val = 0;
        for (int i = 0; i < 16; i++) {
            int x = v * i + 3;
            if (x > 50) {
                found = i;
                found_val = x;
                break;
            }
        }
        out[tid * 2 + 0] = found;
        out[tid * 2 + 1] = found_val;
    }
}

// ------------------------------------------------------------------
// Multiple exits with different values.

__global__ void multi_exit_val(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        if (v < 0) {
            result = -1;
        } else if (v == 0) {
            result = 0;
        } else if (v < 10) {
            result = v * v;
        } else if (v < 100) {
            result = v + 100;
        } else {
            result = 999;
        }
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Loop with two loop-carried variables.

__global__ void two_carry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v, b = v + 1;
        for (int i = 0; i < 8; i++) {
            int na = a + b;
            int nb = a - b + i;
            a = na;
            b = nb;
        }
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}

// ------------------------------------------------------------------
// Nested loop with outer-carried variable updated in inner loop.

__global__ void nested_carry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int total = 0;
        for (int i = 1; i <= 4; i++) {
            int row = 0;
            for (int j = 1; j <= i; j++) {
                row += v * j;
            }
            total += row;
        }
        out[tid] = total;
    }
}

// ------------------------------------------------------------------
// Short-circuit with side effects: tests CFG correctness.
// Both sides of && compute a value; only the second is conditional.

__global__ void short_circ_val(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = 0;
        // The 100/v division must not execute when v==0
        if (v != 0 && 100 / v > 5) {
            x = 1;
        }
        // The table[v] access must not execute when v<0 or v>=16
        int table[16];
        for (int i = 0; i < 16; i++) table[i] = i * 3;
        int y = 0;
        if (v >= 0 && v < 16 && table[v] > 20) {
            y = table[v];
        }
        out[tid] = x + y;
    }
}

// ------------------------------------------------------------------
// Boolean accumulated across iterations.

__global__ void bool_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int all_pos = 1;
        int any_neg = 0;
        int count_zero = 0;
        for (int i = 0; i < 8; i++) {
            int x = v + i * 3 - 10;
            if (x <= 0) all_pos = 0;
            if (x < 0) any_neg = 1;
            if (x == 0) count_zero++;
        }
        out[tid] = all_pos * 100 + any_neg * 10 + count_zero;
    }
}

// ------------------------------------------------------------------
// Phi through multiple if/else branches: all paths assign different exprs.

__global__ void if_else_phi(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        int r;
        if (av > bv && av > cv) {
            r = av * 2;
        } else if (bv > av && bv > cv) {
            r = bv * 3;
        } else if (cv > av && cv > bv) {
            r = cv * 5;
        } else {
            r = av + bv + cv;
        }
        out[tid] = r;
    }
}
