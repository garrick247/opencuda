// Probe: correctness-focused — verify computed values match expected results.
// Test patterns where silent wrong-code generation is more likely:
// - Loop-carried accumulator with conditional skip vs unconditional add
// - Ternary producing one of two array loads
// - Post-increment vs pre-increment in same expression (no UB — separate stmts)
// - Integer overflow at boundary (INT_MIN/-1, INT_MAX+1)
// - Float precision: 1.0f/3.0f * 3.0f != 1.0f
// - Short-circuit && evaluation
// - Compound += in loop exiting early

// ------------------------------------------------------------------
// Loop-carried with conditional skip (the inner add happens only on odd i).

__global__ void cond_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 1) s += in[i];
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Ternary selecting one of two array loads.

__global__ void ternary_load(int *out, int *a, int *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Ternary on the address, then dereference
        int r = (sel[tid] & 1) ? a[tid] : b[tid];
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Pre-increment vs post-increment in separate statements (no UB).

__global__ void inc_forms(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v++;    // a = v,   v = v+1
        int b = ++v;    // v = v+1, b = v
        // a = original, v = original+2, b = original+2
        out[tid] = a + b + v;  // = orig + (orig+2) + (orig+2) = 3*orig+4
    }
}

// ------------------------------------------------------------------
// Short-circuit &&: second operand should NOT be evaluated if first is false.

__device__ int always_true_counter;  // not actually used in logic

__device__ int safe_div(int *flag, int a, int b) {
    // Only divide if b != 0
    return (b != 0 && a / b > 0) ? 1 : 0;
}

__global__ void short_circuit_and(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = safe_div(0, a[tid], b[tid]);
}

// ------------------------------------------------------------------
// Compound += in loop with early break.

__global__ void compound_early_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < n; i++) {
            s += in[i];
            if (s > 100) break;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Integer boundary: -2147483648 / -1 should clamp or be handled (UB in C
// but PTX div handles it); and INT_MAX + 1 wraps.

__global__ void int_boundary(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int max_val = 2147483647;
        int wrapped = max_val + 1;   // wraps to INT_MIN in 2's complement
        out[tid] = wrapped;           // should be -2147483648
    }
}

// ------------------------------------------------------------------
// Float precision accumulation.

__global__ void float_accum_order(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        // Sum 1/3 exactly n times: result will not be exactly n/3
        for (int i = 0; i < 8; i++) s += 1.0f / 3.0f;
        out[tid] = s;  // will be ~2.666... (8/3), not exactly 2.666...
    }
}

// ------------------------------------------------------------------
// Boolean logic: (a || b) && !(a && b) — XOR via boolean algebra.

__global__ void bool_xor(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid] != 0;
        int y = b[tid] != 0;
        out[tid] = (x || y) && !(x && y);  // true only when exactly one is nonzero
    }
}

// ------------------------------------------------------------------
// Array-of-arrays: int lut[4][4] where row and col are both runtime.

__global__ void arr_of_arr(int *out, int *row_in, int *col_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lut[4][4] = {
            {1,  2,  3,  4},
            {5,  6,  7,  8},
            {9, 10, 11, 12},
            {13,14, 15, 16}
        };
        int r = row_in[tid] & 3;
        int c = col_in[tid] & 3;
        out[tid] = lut[r][c];
    }
}

// ------------------------------------------------------------------
// Swap via temporary (classic idiom).

__device__ void swap_int(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void swap_kernel(int *x, int *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) swap_int(&x[tid], &y[tid]);
}
