// Probe: type cast chains, implicit narrowing/widening, C-style cast
// patterns, and expression type coercion edge cases.

// ------------------------------------------------------------------
// Cast chain: float → int → long long → float.

__global__ void cast_chain(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = in[tid];
        int   i = (int)f;
        long long ll = (long long)i;
        float f2 = (float)ll;
        out[tid] = f2 + (float)i + (float)ll;
    }
}

// ------------------------------------------------------------------
// Unsigned cast preserving bit pattern.

__global__ void uint_cast(unsigned int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned int u = (unsigned int)v;  // reinterpret, no sign ext
        unsigned int shifted = u >> 1;
        out[tid] = shifted | (u & 1u);
    }
}

// ------------------------------------------------------------------
// Char/short arithmetic with implicit widening.

__global__ void narrow_arith(int *out, signed char *a, short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = (int)a[tid];  // sign-extend
        int bv = (int)b[tid];  // sign-extend
        out[tid] = av + bv + av * bv;
    }
}

// ------------------------------------------------------------------
// Double intermediate precision.

__global__ void double_precision(double *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = (double)in[tid];
        double r = v * v + 2.0 * v + 1.0;  // (v+1)^2 in double
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Mixed double/float — result type follows the widest operand.

__global__ void mixed_double_float(double *out, float *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float  fa = a[tid];
        double db = b[tid];
        double r = fa + db;  // float + double → double
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Implicit bool-to-int in arithmetic.

__global__ void bool_arith2(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        int r = (av > 0) + (bv > 0) + (av > bv) + (av == bv) + (av < bv);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Accumulate float comparisons as float: (condition) ? 1.0f : 0.0f.

__global__ void float_cond_sum(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid];
        float r = (av > bv) ? 1.0f : 0.0f;
        r += (av > 0.0f) ? 0.5f : 0.0f;
        r += (bv < 0.0f) ? 0.25f : 0.0f;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Integer to pointer cast and back (ptrdiff-style arithmetic).

__global__ void ptr_diff(int *out, int *base, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = base + tid;
        int *q = base + (n - 1 - tid);
        // Pointer difference in units of elements
        long long diff = (long long)(p - q);
        out[tid] = (int)(diff < 0 ? -diff : diff);
    }
}

// ------------------------------------------------------------------
// Truncating cast: long long → int (low 32 bits).

__global__ void ll_to_int(int *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        int lo = (int)(v & 0xFFFFFFFFLL);
        int hi = (int)(v >> 32);
        out[tid * 2 + 0] = lo;
        out[tid * 2 + 1] = hi;
    }
}
