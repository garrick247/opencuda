// Regression: logical NOT of predicate and scientific notation float literals.
//
// Without fix 1 (not.pred):
//   int s = !(v > 0);  → setp.eq.s32 %p_dest, %p_src, 0
//   where %p_src is a predicate register → ptxas "Arguments mismatch"
//   Fix: emit not.pred %p_dest, %p_src for ! applied to a predicate.
//
// Without fix 2 (sci_notation):
//   float eps = 1e-6f; → ParseError "expected SEMI, got IDENT 'e'"
//   Fix: FLOAT_LIT regex now handles [eE][+-]?[0-9]+ exponent suffix.

__global__ void logical_not_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = !v;              // !int → setp.eq(v, 0)
        int s = !(v > 0);        // !(pred) → not.pred
        int t = !(!v);           // double negation
        int u = !(v == 0);       // ! of eq-comparison predicate
        out[tid] = r + s + t + u;
    }
}

#define EPS 1e-6f
#define SMALL 1.5e-3
#define BIG 6.02e23f

__global__ void sci_notation_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float eps = EPS;
        float small = (float)SMALL;
        // Scientific notation literals in expressions
        float threshold = 1e-4f;
        float scale = 2.5e2f;
        int near_zero = (v > -eps && v < eps) ? 1 : 0;
        out[tid] = near_zero ? small : (v * scale + threshold);
    }
}

__global__ void double_sci_test(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double eps = 1e-12;
        double scale = 6.674e-11;  // gravitational constant
        out[tid] = v * scale + eps;
    }
}
