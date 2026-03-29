// Probe: complex CFG patterns — nested short-circuit in loop condition,
// phi nodes in deeply nested branches, continue/break from deep nesting,
// and early-exit patterns that stress dominance analysis.

// ------------------------------------------------------------------
// Nested && in while condition with loop-carried variable.

__global__ void nested_and_while(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0, i = 0;
        // Compound condition uses both runtime and loop-carried values
        while (i < 16 && acc < 100 && v > 0) {
            acc += v;
            v--;
            i++;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Triple nested loops with inner-most continue/break.

__global__ void triple_nested_flow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        int v = in[tid];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (k == 2) continue;   // innermost continue
                    if (j == 3) break;       // inner break
                    acc += i * j + k;
                }
                if (i == 2 && j == 1) break;  // break from j-loop
            }
        }
        out[tid] = acc + v;
    }
}

// ------------------------------------------------------------------
// Short-circuit || in if condition with variable assigned in RHS.

__global__ void short_circuit_or_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = -1;
        // x assigned in RHS of ||, but LHS might short-circuit it
        if (v < 0 || (x = v * 2) > 100) {
            out[tid] = x + 1;  // x might be -1 (if v < 0) or v*2
        } else {
            out[tid] = x;      // x = v*2 (LHS was false)
        }
    }
}

// ------------------------------------------------------------------
// Boolean accumulation in loop with early exit.

__global__ void bool_accum_early_exit(int *out, int *in, int n, int m) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Count how many elements satisfy condition, stop at 4
        int count = 0;
        for (int i = tid; i < m && count < 4; i += n) {
            if (in[i] > 0) {
                count++;
            }
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Do-while with complex condition.

__global__ void do_while_complex(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0, acc = 0;
        do {
            acc += v;
            v = (v > 0) ? v - 1 : v + 1;  // approach zero
            i++;
        } while (v != 0 && i < 20);
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Phi in complex switch — switch inside loop.

__global__ void switch_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            int x;
            switch ((v + i) % 4) {
                case 0: x = i;        break;
                case 1: x = i * 2;    break;
                case 2: x = i + v;    break;
                default: x = v - i;   break;
            }
            acc += x;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Complex CFG: if inside switch inside loop inside if.

__global__ void deep_cfg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        if (v > 0) {
            for (int i = 0; i < 4; i++) {
                switch (i) {
                    case 0:
                        if (v > 100) result += 10;
                        else result += 1;
                        break;
                    case 1:
                        result += v > 50 ? 5 : 2;
                        break;
                    default:
                        result += i;
                }
            }
        }
        out[tid] = result;
    }
}
