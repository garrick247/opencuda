// Probe: predicate/setp edge cases — predicates stored to memory,
// used as function arguments, chained in complex boolean trees,
// and converted to int in various contexts.

// ------------------------------------------------------------------
// Predicate (bool) stored in array.

__global__ void pred_store(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Store booleans as 0/1 integers
        out[tid * 4 + 0] = (v > 0);
        out[tid * 4 + 1] = (v == 0);
        out[tid * 4 + 2] = (v < 0);
        out[tid * 4 + 3] = (v >= -10 && v <= 10);
    }
}

// ------------------------------------------------------------------
// Boolean expression as function argument.

__device__ int bool_arg(int a, int b, int c) {
    return a * 100 + b * 10 + c;
}

__global__ void bool_as_arg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = bool_arg(v > 100, v > 50, v > 0);
    }
}

// ------------------------------------------------------------------
// Boolean in ternary: (cond1 && cond2) ? x : y.

__global__ void bool_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Complex condition in ternary
        int r = (v > 0 && v < 100 && v % 2 == 0) ? v * 2 : -v;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Boolean arithmetic: multiply boolean by value.

__global__ void bool_multiply(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // branchless abs
        int sign = (v < 0);   // 1 if negative, 0 if non-negative
        int abs_v = (1 - 2 * sign) * v;
        out[tid] = abs_v;
    }
}

// ------------------------------------------------------------------
// OR of multiple comparisons used in if.

__global__ void multi_cmp_or(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v == 0 || v == 1 || v == 2 || v == -1 || v == -2) {
            out[tid] = 1;
        } else {
            out[tid] = 0;
        }
    }
}

// ------------------------------------------------------------------
// AND of multiple comparisons in loop condition.

__global__ void multi_cmp_and_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int y = in[tid] + 1;
        int z = in[tid] - 1;
        int count = 0;
        while (x > 0 && y > 0 && z > -100 && count < 20) {
            x -= 3;
            y -= 2;
            z -= 1;
            count++;
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Negated boolean expression.

__global__ void bool_negate(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int in_range = (v >= -50 && v <= 50);
        int out_of_range = !in_range;
        int not_zero = (v != 0);
        int is_zero = !not_zero;
        out[tid] = in_range + out_of_range + not_zero + is_zero;  // always == 2
    }
}

// ------------------------------------------------------------------
// Boolean XOR used as "exactly one of".

__global__ void bool_xor(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int pa = (a[tid] > 0);
        int pb = (b[tid] > 0);
        out[tid] = pa ^ pb;  // 1 if exactly one is positive
    }
}

// ------------------------------------------------------------------
// Boolean result from comparison, then compared again.

__global__ void cmp_of_cmp2(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int pos = (v > 0);         // 0 or 1
        int very_pos = (v > 100);  // 0 or 1
        // Compare the two booleans
        int both = (pos == very_pos);  // 1 if same, 0 if different
        out[tid] = both;
    }
}
