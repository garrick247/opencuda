// Probe: scoping, shadowing, and complex statement forms that
// exercise the parser's variable resolution.

// ------------------------------------------------------------------
// Inner scope shadows outer variable.

__global__ void scope_shadow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    int result = v;
    {
        int v = result * 2;  // shadows outer v
        result = v + 1;
    }
    // Outer v is still in scope here
    out[tid] = result + v;
}

// ------------------------------------------------------------------
// For-loop variable scoped to loop.

__global__ void for_scope(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            int v = in[tid] + i;  // v scoped to loop body
            sum += v;
        }
        // i and v no longer in scope
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Multiple scopes within same function.

__global__ void multi_scope(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = 0;
        {
            int x = in[tid];
            r += x;
        }
        {
            int x = in[tid] * 2;  // different x in different scope
            r += x;
        }
        out[tid] = r;  // r = 3 * in[tid]
    }
}

// ------------------------------------------------------------------
// Function parameter name that would shadow outer.

__device__ int shadow_param(int x, int y) {
    int x2 = x * x;  // no shadowing — x is param
    return x2 + y;
}

__global__ void shadow_call(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = shadow_param(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Complex loop: break in middle of computed expression.

__global__ void break_mid_expr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int i = 0;
        while (1) {
            int delta = v * i + 1;
            if (delta > 100) break;
            acc += delta;
            i++;
            if (i >= 20) break;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Switch with complex case expressions.

__global__ void switch_complex(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        switch (v & 7) {
            case 0: r = v; break;
            case 1: r = v + 1; break;
            case 2: r = v + 2; break;
            case 3: r = v + 3; break;
            case 4: r = v * 2; break;
            case 5: r = v * 3; break;
            case 6: r = v * 4; break;
            case 7: r = v * v; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// If-else chain resulting in same variable being assigned.

__global__ void if_else_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 100) {
            r = v - 100;
        } else if (v > 50) {
            r = v - 50;
        } else if (v > 0) {
            r = v;
        } else if (v > -50) {
            r = -v;
        } else {
            r = 100;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Nested function call as argument.

__device__ int add_one(int x) { return x + 1; }
__device__ int double_it(int x) { return x * 2; }
__device__ int negate(int x) { return -x; }

__global__ void nested_fn_args(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Nested calls: add_one(double_it(negate(in[tid])))
        int r = add_one(double_it(negate(in[tid])));
        out[tid] = r;
    }
}
