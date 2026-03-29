// Probe: pointer arithmetic edge cases, do-while with compound cond,
// prefix increment/decrement, pre/post increment in expressions.

// ------------------------------------------------------------------
// Prefix ++ and -- on loop variables.
// Tests that ++i and --j emit the correct increment/decrement.

__global__ void prefix_incdec(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        while (i < n) {
            sum += i;
            ++i;
        }
        int j = n - 1;
        while (j >= 0) {
            sum += j;
            --j;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Postfix ++ in expression context: out[i++] = val (side effect on i).

__global__ void postfix_in_expr(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        for (int k = 0; k < n; k++) {
            out[i++] = data[k] * 2;
        }
    }
}

// ------------------------------------------------------------------
// do-while with compound condition: do { ... } while (x > 0 && y < max).

__global__ void do_while_compound(int *out, int x, int max_iter) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int y = 0;
        do {
            x -= 2;
            y++;
            count++;
        } while (x > 0 && y < max_iter);
        out[0] = count;
    }
}

// ------------------------------------------------------------------
// Multiple variables in for-loop init: for (int i = 0, j = n-1; i < j; ...)
// Tests that multi-declaration in for-init is parsed correctly.

__global__ void two_ptr_scan(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0, j = n - 1; i < j; i++, j--) {
            sum += data[i] + data[j];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Pointer to pointer: int ** — load pointer, then load again.

__global__ void ptr_to_ptr(int *out, int **ptrs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = *ptrs[tid];
    }
}
