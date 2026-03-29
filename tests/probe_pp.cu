// Probe: register pressure and liveness edge cases.
// Patterns where the naive register allocator might produce wrong output
// due to aliasing, back-edge liveness, or writeback conflicts.

// ------------------------------------------------------------------
// Many live variables simultaneously.
// 8 live scalars across a loop — stresses register window.

__global__ void eight_live(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 1, b = 2, c = 3, d = 4;
        int e = 5, f = 6, g = 7, h = 8;
        for (int i = 0; i < n; i++) {
            int v = data[i];
            a += v; b -= v; c ^= v; d += v * 2;
            e ^= a; f ^= b; g ^= c; h ^= d;
        }
        out[0] = a + b + c + d + e + f + g + h;
    }
}

// ------------------------------------------------------------------
// Loop-carried value with writeback that aliases loop index.
// `for (int i = 0; i < n; i++) { sum += i * data[i]; }`
// The `sum` writeback must not clobber `i`.

__global__ void index_multiply_sum(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += i * data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Nested loops with separate loop-carried values.
// Inner and outer induction vars must be independent.

__global__ void nested_loops_liveness(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            int inner_sum = 0;
            for (int j = 0; j <= i; j++) {
                inner_sum += data[j];
            }
            total += inner_sum;
        }
        out[0] = total;
    }
}

// ------------------------------------------------------------------
// Value used after a function call that could alias registers.
// `a = data[0]; b = fn(a, x); out[0] = a + b;` — `a` must survive fn call.

__device__ int double_and_add(int x, int y) {
    return x * 2 + y;
}

__global__ void value_after_call(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = data[tid];
        int b = double_and_add(a, tid);
        out[tid] = a + b;  // `a` must be live across the call
    }
}

// ------------------------------------------------------------------
// Loop with conditional writeback to multiple variables.
// Tests that different writeback paths don't corrupt each other.

__global__ void conditional_writeback(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_sum = 0, neg_sum = 0, count = 0;
        for (int i = 0; i < n; i++) {
            int v = data[i];
            if (v > 0) {
                pos_sum += v;
                count++;
            } else if (v < 0) {
                neg_sum += v;
                count++;
            }
        }
        out[0] = pos_sum + neg_sum + count;
    }
}

// ------------------------------------------------------------------
// Short-circuit inside loop with live variable after merge.
// The `max_val` var must remain live across the short-circuit blocks.

__global__ void sc_with_live_across(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int max_val = -2147483648;
        int i = 0;
        while (i < n && data[i] != -1) {
            if (data[i] > max_val) max_val = data[i];
            i++;
        }
        out[0] = max_val;
    }
}

// ------------------------------------------------------------------
// Variable reused after loop exit — tests that loop-carried writeback
// doesn't clobber the variable in the exit path.

__global__ void reuse_after_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int last = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
            last = data[i];
        }
        // sum and last must be correct after loop
        out[0] = sum * last;
    }
}
