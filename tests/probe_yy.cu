// Probe: adversarial C patterns — array subscript with side-effect expression,
// comma expression in various contexts, cast-then-deref ((int*)ptr)[i],
// assignment inside while-condition, nested function calls f(g(h(x))),
// expression statement with no side effect (dead expression), bitwise
// NOT (~) on various types, left-shift overflow, and array-of-pointers.

// ------------------------------------------------------------------
// Array subscript with complex index expression.

__global__ void complex_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Index is an expression: (tid * 3 + 1) & (n - 1)
        int idx = (tid * 3 + 1) & (n - 1);
        out[tid] = in[idx];
    }
}

// ------------------------------------------------------------------
// Cast-then-deref: treat int array as char array.

__global__ void cast_deref(unsigned char *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Cast int pointer to char pointer, read byte 0
        unsigned char *bp = (unsigned char *)&in[tid];
        out[tid] = bp[0];  // lowest byte of in[tid]
    }
}

// ------------------------------------------------------------------
// Nested function calls: f(g(h(x)))

__device__ int inc(int x)    { return x + 1; }
__device__ int dbl(int x)    { return x * 2; }
__device__ int sq(int x)     { return x * x; }

__global__ void nested_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sq(dbl(inc(in[tid]))) = (2*(in[tid]+1))^2
        out[tid] = sq(dbl(inc(in[tid])));
    }
}

// ------------------------------------------------------------------
// Bitwise NOT (~) on int, unsigned, and char.

__global__ void bitwise_not(int *out_i, unsigned *out_u, int *in_i, unsigned *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_i[tid] = ~in_i[tid];
        out_u[tid] = ~in_u[tid];
    }
}

// ------------------------------------------------------------------
// Left shift that overflows (well-defined for unsigned).

__global__ void shift_overflow(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned v = in[tid];
        // Shift by amount modulo 32 (PTX behavior)
        unsigned r1 = v << 16;     // high bits overflow
        unsigned r2 = v << 31;     // only LSB survives
        out[tid * 2]     = r1;
        out[tid * 2 + 1] = r2;
    }
}

// ------------------------------------------------------------------
// Multiple return types from device functions called in sequence.

__device__ float float_fn(float x) { return x * 1.5f; }
__device__ int int_fn(int x)       { return x + 7; }
__device__ double double_fn_d(double x) { return x * 2.0; }

__global__ void mixed_return_types(float *out_f, int *out_i, double *out_d,
                                      float *in_f, int *in_i, double *in_d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_f[tid] = float_fn(in_f[tid]);
        out_i[tid] = int_fn(in_i[tid]);
        out_d[tid] = double_fn_d(in_d[tid]);
    }
}

// ------------------------------------------------------------------
// Pointer subtraction (ptrdiff).

__global__ void ptr_diff(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Distance between two pointers (in elements)
        int *pa = a + tid;
        int *pb = b + tid;
        long long diff = (long long)pb - (long long)pa;
        out[tid] = (int)(diff / (long long)sizeof(int));
    }
}

// ------------------------------------------------------------------
// Ternary as lvalue-selector through pointers (write to one of two targets).

__global__ void ternary_target(int *out_even, int *out_odd, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int *target = (tid % 2 == 0) ? &out_even[tid] : &out_odd[tid];
        *target = v * v;
    }
}

// ------------------------------------------------------------------
// Chain of comparisons producing a score.

__device__ int score(int a, int b, int c, int d) {
    int s = 0;
    if (a > b) s++;
    if (a > c) s++;
    if (a > d) s++;
    if (b > c) s++;
    if (b > d) s++;
    if (c > d) s++;
    return s;
}

__global__ void score_kernel(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = score(a[tid], b[tid], c[tid], d[tid]);
}

// ------------------------------------------------------------------
// Explicit zero-init of struct via field assignment.

struct RGBA { int r, g, b, a; };

__global__ void struct_zero(struct RGBA *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct RGBA px;
        px.r = 0; px.g = 0; px.b = 0; px.a = 255;
        out[tid] = px;
    }
}

// ------------------------------------------------------------------
// Recursive-like pattern via loop unrolling (simulates 4-deep recursion).

__device__ int collatz_steps(int v) {
    int s = 0;
    while (v > 1 && s < 100) {
        if (v % 2 == 0) v /= 2;
        else v = 3*v + 1;
        s++;
    }
    return s;
}

__global__ void collatz_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = collatz_steps(in[tid]);
}
