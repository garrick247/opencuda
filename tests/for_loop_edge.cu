// Regression: for-loop with empty condition and/or empty increment
// Without fix:
//   - empty condition 'for(;;)': ParseError "unexpected token ';'"
//   - empty increment 'for(init; cond;)': ParseError "unexpected token ')'"
// Fix: _parse_stmt for-loop handler checks for empty cond/inc before
//      calling _parse_expr(), substituting Const(1) for empty condition.

// Empty increment: for(init; cond;) body
__global__ void for_no_inc(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        int i = 0;
        for (; i < 4;) {
            val = val * 0.5f + in[(tid + i) % n];
            i++;
        }
        out[tid] = val;
    }
}

// Empty body with semicolon: for(init; cond; inc) ;
__global__ void for_empty_body(int *out, int n) {
    int tid = threadIdx.x;
    int count = 0;
    for (; count < tid && count < n; count++)
        ;
    out[tid] = count;
}

// Infinite loop with break: for(;;) { ... break; }
__global__ void for_inf_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int steps = 0;
        for (;;) {
            if (v <= 1) break;
            v = (v % 2 == 0) ? v / 2 : v * 3 + 1;
            steps++;
            if (steps >= 100) break;
        }
        out[tid] = steps;
    }
}

// Multi-init, multi-update: for(int i=0, j=n-1; i<j; i++, j--)
__global__ void for_multi_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0, j = n - 1; i <= j; i++, j--) {
            sum += in[i] + in[j];
        }
        out[tid] = sum;
    }
}
