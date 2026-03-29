// Probe: optimizer safety — values that must NOT be hoisted, folded,
// or CSE'd across block boundaries, plus ternary-with-side-effects.

// ------------------------------------------------------------------
// Loop: invariant expression in loop, but memory-dependent operand
// that must not be hoisted even if the address is invariant.

__global__ void no_hoist_volatile_read(int *out, volatile int *flag, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            // flag[0] is volatile — cannot hoist even if address is invariant
            int f = flag[0];
            acc += i * f;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Two loads from same address in different blocks — must NOT CSE.

__global__ void no_cse_across_blocks(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 0) {
            int a = in[tid];   // same address as v above, but in different block
            r = a + 1;
        } else {
            int b = in[tid];   // same address again
            r = b - 1;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Ternary with function call in condition (condition has a side effect).

__global__ void ternary_call_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // The atomicAdd has a side effect — must not be CSE'd away
        int old = atomicAdd(&out[0], 1);
        int r = (old > 5) ? v : -v;
        out[tid + 1] = r;
    }
}

// ------------------------------------------------------------------
// Constant fold boundary: INT_MAX + 1 should NOT fold incorrectly.

__global__ void const_fold_boundary(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // These should fold to known constants
        int a = 2147483647;      // INT_MAX
        int b = -2147483648;     // INT_MIN
        int c = 2147483647 / 2;  // folds to 1073741823
        int d = 2147483646;      // one below INT_MAX
        out[tid] = (a > b) ? c + d : c - d;
    }
}

// ------------------------------------------------------------------
// Dead store that IS a side effect (volatile store — must not eliminate).

__global__ void no_dead_volatile(volatile int *shared_flag, int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // This volatile store must not be eliminated even though
        // shared_flag is not read back in this thread
        shared_flag[tid] = tid;
        out[tid] = tid * 2;
    }
}

// ------------------------------------------------------------------
// LICM safety: loop-invariant COMPARISON using loop-carried value.
// The comparison must NOT be hoisted — it uses loop-carried v.

__global__ void no_hoist_loop_carried_cmp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            // v is loop-invariant BUT using it in a comparison that
            // controls a runtime branch might get tricky
            int x = (v > i) ? v - i : i - v;  // abs(v - i) via ternary
            acc += x;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Multiple assigns to same var in different arms — must merge correctly.

__global__ void multi_assign_same_var(float *out, float *a, float *b,
                                       float *c, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float r;
        int s = sel[tid];
        if (s == 0) {
            r = a[tid];
            r = r * 2.0f;    // second assign in same arm
            r = r + 1.0f;    // third
        } else if (s == 1) {
            r = b[tid] - 1.0f;
        } else {
            r = c[tid] + c[tid];
        }
        out[tid] = r;
    }
}
