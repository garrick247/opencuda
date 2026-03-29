// Probe: short-circuit edge cases not yet covered.
// - Bare || as while condition (no outer &&)
// - Triple || chaining: a || b || c
// - Mixed: (a && b) || (c && d)
// - Logical NOT on || result: !(a || b)
// - Multiple independent short-circuit ifs in same kernel (label collision)
// - Short-circuit inside ternary *branches* (not the ternary condition itself)
// - && where LHS is an integer register (not a direct comparison)

// ------------------------------------------------------------------
// Bare || as while-loop condition.
// `while (a[i] != 0 || b[i] != 0)` — no outer &&, just ||.

__global__ void bare_or_while(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int i = 0;
        while (i < n && (a[i] != 0 || b[i] != 0)) {
            count++;
            i++;
        }
        out[0] = count;
    }
}

// ------------------------------------------------------------------
// Triple || chain: a || b || c — tests that lor_rhs/lor_skip/lor_merge
// labels don't collide between the two || operators.

__global__ void triple_or(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid];
        out[tid] = (va > 0 || vb > 0 || vc > 0) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Mixed: (a && b) || (c && d).
// LHS short-circuits to true → skip RHS entirely.
// LHS short-circuits to false → evaluate RHS's own &&.

__global__ void mixed_and_or(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid], vd = d[tid];
        int result = (va > 0 && vb > 0) || (vc > 0 && vd > 0);
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Logical NOT on || result: !(a[tid] > 0 || b[tid] > 0).
// Tests that unary ! correctly inverts the multi-block short-circuit result.

__global__ void not_or(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = !(a[tid] > 0 || b[tid] > 0) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Multiple independent short-circuit ifs in same kernel.
// Tests that land_rhs/land_skip/land_merge labels are unique across
// different if statements in the same function.

__global__ void multi_sc_ifs(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r = 0;
        // First short-circuit if
        if (v > 0 && v < 50) {
            r += 1;
        }
        // Second short-circuit if (must get distinct block labels)
        if (v > 25 && v < 75) {
            r += 2;
        }
        // Third short-circuit if
        if (v > 50 && v < 100) {
            r += 4;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Short-circuit inside ternary *branches* (not the ternary condition).
// `x ? (a && b) : (c || d)` — the && and || are in the true/false arms.

__global__ void sc_in_ternary_arms(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid], vd = d[tid];
        // pivot on a[tid] > 0: if true evaluate && else evaluate ||
        int result = (va > 0) ? (vb > 0 && vc > 0) : (vc > 0 || vd > 0);
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// && where the LHS is an integer variable (not a direct comparison).
// `int flag = (v > 0); if (flag && v < 100)` — LHS is a register int,
// not a fresh CmpInst. Tests that CondBrTerm accepts non-predicate LHS.

__global__ void and_with_int_lhs(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int flag = (v > 0);   // flag is 0 or 1 (integer, not predicate)
        int result = 0;
        if (flag && v < 100) {
            result = v;
        }
        out[tid] = result;
    }
}
