// Probe: edge cases in statements and expressions — unary +, comma
// operator in for, for with empty body, conditional without braces,
// and very minimal kernels.

// ------------------------------------------------------------------
// No-parameter kernel.

__global__ void no_params() {
    // Intentionally empty — just tests that it compiles
}

// ------------------------------------------------------------------
// Unary + operator.

__global__ void unary_plus(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = +v;        // unary plus — same as v
        int b = +(v * 2);  // unary plus on expression
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// For loop with empty body.

__global__ void for_empty_body(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int count = 0;
        // Count leading zeros (simplified)
        for (int i = 31; i >= 0 && !((tid + 1) & (1 << i)); i--)
            count++;
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// If without braces.

__global__ void if_no_braces(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        if (v > 0)
            r = 1;
        else if (v < 0)
            r = -1;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// While without braces.

__global__ void while_no_braces(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        while (v > 0 && count < 20)
            v -= 3, count++;
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Comma operator in for update.

__global__ void for_comma_update(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0, j = 10;
        int acc = 0;
        for (; i < 5; i++, j--) {
            acc += i + j;
        }
        out[tid] = acc + tid;
    }
}

// ------------------------------------------------------------------
// Multiple init in for.

__global__ void for_multi_init(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int i, j;
        for (i = 0, j = v; i < 4 && j > 0; i++, j--) {
            acc += i * j;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Nested if without braces (dangling else).

__global__ void nested_if_no_brace(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 0)
            if (v > 50)
                r = 2;
            else
                r = 1;
        else
            r = 0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Semicolon as empty statement.

__global__ void empty_stmt(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        ;                  // empty statement
        int r = v + 1;
        ;;                 // two empty statements
        out[tid] = r;
    }
}
