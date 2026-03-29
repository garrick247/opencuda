// Probe: very complex control flow — multiple loop exits feeding the
// same phi node, value live across multiple non-adjacent blocks,
// and deeply nested conditionals inside loops.

// ------------------------------------------------------------------
// Multiple break paths, value from last iteration.

__global__ void last_iter_val(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int last_i = -1;
        int last_x = 0;
        for (int i = 0; i < 16; i++) {
            int x = v * (i + 1) - 10;
            if (x > 50) break;
            if (x < 0) {
                last_i = -i;
                last_x = x;
                continue;
            }
            last_i = i;
            last_x = x;
        }
        out[tid * 2 + 0] = last_i;
        out[tid * 2 + 1] = last_x;
    }
}

// ------------------------------------------------------------------
// Value modified in some branches of if/else ladder.

__global__ void partial_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 10, b = 20, c = 30;
        if (v > 50) {
            a = v;
        } else if (v > 25) {
            b = v;
        } else if (v > 0) {
            c = v;
        }
        // a, b, c use phi — each was updated in exactly one path
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Double nested loop with inner early exit and outer accumulation.

__global__ void double_nested_exit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int total = 0;
        for (int i = 0; i < 4; i++) {
            int sub = 0;
            for (int j = 0; j < 4; j++) {
                int val = v * i + j;
                if (val > 20) goto next_i;  // C goto: jump to outer continue
                sub += val;
            }
            total += sub;
            continue;
        next_i:
            total += sub * 2;  // penalty for early exit
        }
        out[tid] = total;
    }
}

// ------------------------------------------------------------------
// Loop with complex condition involving multiple variables.

__global__ void complex_while_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = v, y = v + 1, z = 0;
        while (x > 0 && y > 0 && z < 20) {
            x -= 2;
            y -= 1;
            z++;
        }
        out[tid] = x + y + z;
    }
}

// ------------------------------------------------------------------
// Fibonacci with loop limit and max tracking.

__global__ void fib_tracked(int *out, int *seeds, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = seeds[tid * 2 + 0];
        int b = seeds[tid * 2 + 1];
        int max_val = (a > b) ? a : b;
        int count = 0;
        while (b < 1000 && count < 30) {
            int c = a + b;
            a = b;
            b = c;
            if (c > max_val) max_val = c;
            count++;
        }
        out[tid * 3 + 0] = b;
        out[tid * 3 + 1] = max_val;
        out[tid * 3 + 2] = count;
    }
}

// ------------------------------------------------------------------
// Value live across non-adjacent blocks.

__global__ void cross_block_val(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = v * 3;  // computed early

        // Branch that doesn't touch x
        int y = 0;
        if (v > 0) {
            y = v + 1;
        } else {
            y = -v + 1;
        }

        // Another branch that doesn't touch x
        int z = 0;
        for (int i = 0; i < 4; i++) {
            z += y + i;
        }

        // x is finally used here, must survive both branches
        out[tid] = x + z;
    }
}

// ------------------------------------------------------------------
// Three-level nested conditional.

__global__ void triple_nested(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        if (v > 0) {
            if (v > 10) {
                if (v > 100) {
                    r = 3;
                } else {
                    r = 2;
                }
            } else {
                r = 1;
            }
        } else {
            if (v < -10) {
                r = -2;
            } else {
                r = -1;
            }
        }
        out[tid] = r;
    }
}
