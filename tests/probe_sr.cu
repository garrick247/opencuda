// Probe: special PTX patterns — predicate-from-bool-arithmetic,
// complex setp chains, predicate register pressure, and the
// critical "predicate used as integer in multiplication" pattern.

// ------------------------------------------------------------------
// Boolean multiplication chain (the v1.22 regression family).

__global__ void bool_mul_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Each comparison yields 0 or 1 as int
        int b0 = (v > 0);
        int b1 = (v > 10);
        int b2 = (v > 100);
        int b3 = (v < 0);
        int b4 = (v < -10);
        // Multiply booleans together
        int r = b0 * 1 + b1 * 2 + b2 * 4 + b3 * 8 + b4 * 16;
        // Use boolean as multiplier
        float fv = (float)v;
        float fr = fv * (float)(v > 0);
        out[tid] = r + (int)fr;
    }
}

// ------------------------------------------------------------------
// Boolean in complex expression tree.

__global__ void bool_expr_tree(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        // Tree of boolean operations
        int p1 = (av > bv);
        int p2 = (bv > cv);
        int p3 = (av > cv);
        int p4 = p1 & p2;       // AND of booleans
        int p5 = p1 | p3;       // OR of booleans
        int p6 = p4 ^ p5;       // XOR
        // Mix into arithmetic
        int r = p1 * av + p2 * bv + p3 * cv + p4 + p5 + p6;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Comparison result used in division.

__global__ void bool_div(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int is_even = (v % 2 == 0);
        // Use boolean as divisor substitute — divide by (is_even ? 2 : 1)
        int divisor = is_even + 1;  // 1 or 2
        out[tid] = v / divisor;
    }
}

// ------------------------------------------------------------------
// Comparison with cast to float, then used in float arithmetic.

__global__ void bool_float_arith(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // (v > 0) cast to float (0.0 or 1.0) used in arithmetic
        float pos = (float)(v > 0.0f);
        float neg = (float)(v < 0.0f);
        float mag = v * pos - v * neg;  // = |v|
        out[tid] = mag + pos - neg;
    }
}

// ------------------------------------------------------------------
// Accumulation of bool*int products across loop.

__global__ void bool_mul_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 1; i <= 16; i++) {
            // (v % i == 0) is a bool, multiplied by i
            acc += (v % i == 0) * i;
        }
        out[tid] = acc;  // sum of all divisors of v up to 16
    }
}

// ------------------------------------------------------------------
// Predicate reuse with different types.

__global__ void pred_mixed_types(int *iout, float *fout, int *in, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int iv = in[tid];
        float fv = fin[tid];
        // Same predicate used for int and float outputs
        int cond = (iv > 0 && fv > 0.0f);
        iout[tid] = cond * iv + (1 - cond) * (-iv);
        fout[tid] = (float)cond * fv + (float)(1 - cond) * (-fv);
    }
}

// ------------------------------------------------------------------
// Boolean used in loop bound.

__global__ void bool_loop_bound(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int limit = (v > 0) * 8 + (v <= 0) * 4;  // 8 if pos, 4 if non-pos
        int acc = 0;
        for (int i = 0; i < limit; i++) {
            acc += i * v;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Comparison stored in array (bool array pattern).

__global__ void bool_array(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int flags[8];
        for (int i = 0; i < 8; i++) {
            flags[i] = (v + i) > 5;
        }
        int count = 0;
        for (int i = 0; i < 8; i++) {
            count += flags[i];
        }
        out[tid] = count;
    }
}
