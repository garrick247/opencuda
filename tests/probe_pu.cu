// Probe: complex control flow — early returns, multi-branch returns,
// nested break/continue, infinite loops with break, switch fall-through.

// ------------------------------------------------------------------
// Early return from nested if.
// Three early-exit paths and one fall-through path.

__global__ void early_exit_nested(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int v = data[tid];
    if (v < 0) {
        out[tid] = -1;
        return;
    }
    if (v == 0) {
        out[tid] = 0;
        return;
    }
    if (v > 1000) {
        out[tid] = 1000;
        return;
    }
    out[tid] = v;
}

// ------------------------------------------------------------------
// Return from inside a loop: find-first positive.
// The loop may exit early via return, or fall through to default.

__global__ void find_first(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            if (data[i] > 0) {
                out[0] = data[i];
                return;
            }
        }
        out[0] = -1;
    }
}

// ------------------------------------------------------------------
// Nested loops with break at outer level via flag.
// Inner loop sets flag → outer loop checks flag → breaks.

__global__ void nested_break_flag(int *out, int *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found = -1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i * cols + j] == 42) {
                    found = i * cols + j;
                    break;
                }
            }
            if (found >= 0) break;
        }
        out[0] = found;
    }
}

// ------------------------------------------------------------------
// Continue in outer loop, break in inner.
// Tests that continue/break target the correct loop level.

__global__ void outer_continue_inner_break(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] < 0) continue;
            int local = 0;
            for (int j = i; j < n; j++) {
                if (data[j] == 0) break;
                local += data[j];
            }
            sum += local;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Do-while with multiple exits.
// Tests that do-while body can break or continue, and condition is checked.

__global__ void do_while_multi_exit(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0, sum = 0;
        do {
            if (i >= n) break;
            int v = data[i];
            i++;
            if (v < 0) continue;
            sum += v;
        } while (i < n);
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// While with condition updated inside body.
// Loop variable modified in multiple places.

__global__ void while_multi_update(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0, sum = 0;
        while (i < n) {
            int v = data[i];
            if (v < 0) {
                i += 2;   // skip two when negative
                continue;
            }
            sum += v;
            i++;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Switch with fall-through and default.
// Cases 1 and 2 fall through to common code.

__global__ void switch_fallthrough(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid] % 4;
        int r = 0;
        switch (v) {
            case 0:
                r = 10;
                break;
            case 1:
            case 2:
                r = 20;  // fall-through from case 1 to case 2
                break;
            case 3:
                r = 30;
                break;
            default:
                r = -1;
                break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// For-loop with complex increment and condition.
// Both condition and increment are non-trivial.

__global__ void complex_for(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0, j = n - 1; i < j; i++, j--) {
            sum += data[i] + data[j];
        }
        out[0] = sum;
    }
}
