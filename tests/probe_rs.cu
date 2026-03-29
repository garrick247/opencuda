// Probe: boolean expressions used as integers (predicate-as-int stress),
// complex nested boolean chains, logical AND/OR short-circuit patterns,
// and comparison result arithmetic.

// ------------------------------------------------------------------
// Boolean arithmetic: use comparison results directly in arithmetic.

__global__ void bool_arith(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // Each comparison produces 0 or 1 as int
        int a = (v > 0);
        int b = (v > 10);
        int c = (v > 100);
        int d = (v < 0);
        // Sum of booleans = count of satisfied conditions
        out[tid] = a + b + c + d;
    }
}

// ------------------------------------------------------------------
// Boolean used in multiplication and ternary.

__global__ void bool_mul(float *out, float *data, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        int f = flags[tid];
        // Conditional scale: multiply by 0 or 1
        float r = v * (float)(f > 0);
        out[tid] = r + (float)(v < 0.0f) * (-v);
    }
}

// ------------------------------------------------------------------
// Nested NOT and AND/OR chains.

__global__ void bool_chains(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        // Complex nested boolean: verify each sub-expression
        int e1 = (av > 0 && bv > 0);
        int e2 = !(av > 0 || bv > 0);
        int e3 = (!e1 && !e2) || (e1 && e2);
        int e4 = (av + bv > cv) ? 1 : 0;
        int e5 = e1 ^ e2;  // XOR of booleans
        out[tid] = e1 + e2 + e3 + e4 + e5;
    }
}

// ------------------------------------------------------------------
// Comparison result stored and compared again.

__global__ void cmp_of_cmp(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int gt0 = (v > 0);    // 0 or 1
        int gt10 = (v > 10);  // 0 or 1
        // Compare booleans
        int same = (gt0 == gt10);
        int differ = (gt0 != gt10);
        out[tid] = same * 10 + differ * 5 + gt0 + gt10;
    }
}

// ------------------------------------------------------------------
// Short-circuit evaluation: && and || with side effects.

__global__ void short_circuit(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r = 0;
        // These should not segfault even when left side is false
        if (v != 0 && (100 / v) > 5) r = 1;
        if (v == 0 || v > 10) r += 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Predicate in array indexing.

__global__ void pred_index(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid];
        // Use boolean as index: 0 or 1
        float table[2];
        table[0] = av * 2.0f;
        table[1] = bv * 3.0f;
        int idx = (av > bv);  // 0 if av <= bv, 1 if av > bv
        out[tid] = table[idx];
    }
}

// ------------------------------------------------------------------
// Predicate stored then used as bool in if.

__global__ void stored_bool(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int is_pos = (v > 0);
        int is_big = (v > 100);
        // Use stored booleans in if
        if (is_pos) {
            out[tid] = is_big ? 2 : 1;
        } else {
            out[tid] = 0;
        }
    }
}
