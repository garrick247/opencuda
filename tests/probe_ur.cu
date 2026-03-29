// Probe: verifier stress test — complex phi patterns, variables
// defined on all paths, partial-def patterns that should pass.

// ------------------------------------------------------------------
// Variable defined in all branches of if-else (should pass verifier).

__global__ void all_branches_defined(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 100) {
            r = v / 2;
        } else if (v > 50) {
            r = v * 2;
        } else if (v > 0) {
            r = v + 10;
        } else if (v > -50) {
            r = v - 10;
        } else {
            r = -v;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Variable assigned in loop, used after loop.

__global__ void loop_then_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int last = 0;
        for (int i = 0; i < 4; i++) {
            last = v * i + last;
        }
        out[tid] = last;
    }
}

// ------------------------------------------------------------------
// Nested if: inner var used only in outer else (dominance safe).

__global__ void nested_scope_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        if (v > 0) {
            int inner = v * 3;
            if (inner > 100) {
                r = inner / 10;
            } else {
                r = inner + 5;
            }
        } else {
            r = -1;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Short-circuit AND: variable assigned in RHS only (should be multi-def).

__global__ void short_circuit_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = 0;
        // x is assigned only if v > 0 (short-circuit AND guard)
        if (v > 0 && (x = v * 2) > 50) {
            out[tid] = x + 1;
        } else {
            out[tid] = x;
        }
    }
}

// ------------------------------------------------------------------
// Loop with break: accumulator must be valid after break path.

__global__ void loop_break_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int found = -1;
        for (int i = 0; i < 8; i++) {
            if (in[i] == v) {
                found = i;
                break;
            }
        }
        out[tid] = found;
    }
}

// ------------------------------------------------------------------
// Ternary as initializer (phi merge at assignment).

__global__ void ternary_init(float *out, float *in, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int f = flags[tid];
        float scale = (f == 0) ? 1.0f : (f == 1) ? 2.0f : 0.5f;
        out[tid] = v * scale;
    }
}

// ------------------------------------------------------------------
// Multiple return paths with same result variable.

__global__ void multi_return_var(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        switch (v % 4) {
            case 0: result = v;     break;
            case 1: result = v + 1; break;
            case 2: result = v + 2; break;
            default: result = v + 3; break;
        }
        out[tid] = result;
    }
}
