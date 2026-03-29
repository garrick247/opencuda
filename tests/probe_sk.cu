// Probe: complex boolean expressions in loop/if conditions,
// predicate reuse across multiple uses, and nested logical ops.

// ------------------------------------------------------------------
// Boolean in loop condition with both && and ||.

__global__ void complex_loop_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0, acc = 0;
        while (i < 16 && (v > 0 || i < 4)) {
            acc += i * v;
            i++;
            if (v < 0) v++;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Predicate used in multiple contexts: arithmetic and branch.

__global__ void pred_multi_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int pos = (v > 0);   // pred → int
        int neg = (v < 0);   // pred → int
        // Used in arithmetic
        int sign = pos - neg;
        // Used in if-branch (as bool)
        if (pos) {
            out[tid] = v + sign;
        } else if (neg) {
            out[tid] = v - sign;
        } else {
            out[tid] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Comparison reused in ternary.

__global__ void cmp_reuse_ternary(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        int gt = (av > bv);
        int ge = (av >= bv);
        // gt and ge used in ternary
        int r1 = gt ? av : bv;
        int r2 = ge ? av : bv;
        out[tid] = r1 + r2 + gt + ge;
    }
}

// ------------------------------------------------------------------
// Chained && short-circuit: verify correct phi at merge.

__global__ void chained_and(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        int r = 0;
        if (av > 0 && bv > 0 && cv > 0) {
            r = 1;
        } else if (av > 0 && bv > 0) {
            r = 2;
        } else if (av > 0) {
            r = 3;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Boolean stored, negated, used in sum.

__global__ void bool_negate_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int b1 = (v > 5);
        int b2 = !(v > 5);   // should be !b1 — same as (v <= 5)
        int b3 = (v >= 0);
        int b4 = !(v >= 0);  // same as (v < 0)
        // b1 + b2 == 1 always; b3 + b4 == 1 always
        out[tid] = b1 + b2 + b3 + b4;  // always 2
    }
}

// ------------------------------------------------------------------
// XOR of predicates.

__global__ void pred_xor(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        int pa = (av > 0);
        int pb = (bv > 0);
        // XOR of boolean values
        int xr = pa ^ pb;
        // Used in branch
        if (xr) {
            out[tid] = av + bv;
        } else {
            out[tid] = av - bv;
        }
    }
}

// ------------------------------------------------------------------
// Boolean accumulation then comparison.

__global__ void bool_then_cmp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        for (int i = 0; i < 8; i++) {
            count += (v + i > 0);
        }
        // Compare the accumulated boolean count
        int r = (count > 4) ? 1 : (count > 2) ? 2 : 3;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Predicate from float comparison in integer context.

__global__ void float_pred_int(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int pos = (v > 0.0f);
        int big = (v > 100.0f);
        int small_abs = (v > -1.0f && v < 1.0f);
        out[tid] = pos * 4 + big * 2 + small_abs;
    }
}
