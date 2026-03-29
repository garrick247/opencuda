// Probe: type system edge cases — bool-to-int coercions, float-to-bool,
// unary operators in expressions, and cast chains.

// ------------------------------------------------------------------
// Boolean result of comparison used in arithmetic (C implicit int cast).
// `int result = (a > b) + (c > d)` — each comparison is 0 or 1.

__global__ void bool_arith(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid], vd = d[tid];
        // Each comparison produces 0 or 1, then we add them
        out[tid] = (va > vb) + (vc > vd);
    }
}

// ------------------------------------------------------------------
// Float comparison result used in integer arithmetic.
// `int npos = (fdata[i] > 0.0f)` — explicit float > 0 → {0,1}.

__global__ void float_bool_arith(int *out, float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int npos = 0;
        for (int i = 0; i < n; i++) {
            npos += (fdata[i] > 0.0f);  // float compare → 0/1 int
        }
        out[0] = npos;
    }
}

// ------------------------------------------------------------------
// Unary operators: unary minus, bitwise NOT, logical NOT.

__global__ void unary_ops(int *out, int *data, float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        float f = fdata[tid];
        int a = -v;          // unary minus on int
        float b = -f;        // unary minus on float
        int c = ~v;          // bitwise NOT
        int d = !v;          // logical NOT: 0 if v!=0, 1 if v==0
        int e = !d;          // double NOT: 0 if d!=0 (v!=0), 1 if d==0 (v==0)
        // e should equal (v != 0)
        out[tid * 5 + 0] = a;
        out[tid * 5 + 1] = (int)b;
        out[tid * 5 + 2] = c;
        out[tid * 5 + 3] = d;
        out[tid * 5 + 4] = e;
    }
}

// ------------------------------------------------------------------
// Cast from bool expression to float: (float)(a > b).
// Tests that the comparison result (0 or 1) is correctly converted to float.

__global__ void bool_to_float(float *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // (a[tid] > b[tid]) gives 0 or 1 as int, then cast to float
        out[tid] = (float)(a[tid] > b[tid]);
    }
}

// ------------------------------------------------------------------
// Conditional expression with mixed int/float types.
// `float r = (v > 0) ? (float)v : -1.0f`

__global__ void cond_mixed_types(float *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        float r = (v > 0) ? (float)v : -1.0f;
        out[tid] = r;
    }
}
