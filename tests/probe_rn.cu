// Probe: sizeof(), enum in switch, #define macros with args, comma
// operator, conditional chains, and integer promotion edge cases.

// ------------------------------------------------------------------
// sizeof(): verify element sizes for various types.

__global__ void sizeof_kernel(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sizeof should return compile-time constants
        int si   = sizeof(int);          // 4
        int sf   = sizeof(float);        // 4
        int sd   = sizeof(double);       // 8
        int sll  = sizeof(long long);    // 8
        int sc   = sizeof(char);         // 1
        int ss   = sizeof(short);        // 2
        out[tid] = si + sf + sd + sll + sc + ss;  // = 27
    }
}

// ------------------------------------------------------------------
// enum in switch.

enum Color { RED = 0, GREEN = 1, BLUE = 2, ALPHA = 3 };

__global__ void enum_switch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int c = in[tid] % 4;
        int r;
        switch (c) {
            case RED:   r = 10; break;
            case GREEN: r = 20; break;
            case BLUE:  r = 30; break;
            default:    r = 40; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// #define macros with arguments.

#define SQ(x)    ((x) * (x))
#define MAX3(a,b,c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

__global__ void macro_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float sq = SQ(v);
        float m = MAX3(v, sq, v + 1.0f);
        float c = CLAMP(m, -100.0f, 100.0f);
        out[tid] = c;
    }
}

// ------------------------------------------------------------------
// Comma operator in for-loop update.

__global__ void comma_for(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum_a = 0, sum_b = 0;
        for (int i = 0, j = n - 1; i < n && j >= 0; i++, j--) {
            sum_a += a[i];
            sum_b += b[j];
        }
        out[tid] = sum_a + sum_b;
    }
}

// ------------------------------------------------------------------
// Conditional expression with side effects: a ? (b=1, b) : (b=2, b).

__global__ void cond_side_effect(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int b = 0;
        int r = (v > 0) ? (b = 1) : (b = -1);
        out[tid] = r + b;
    }
}

// ------------------------------------------------------------------
// Integer promotion: mixed signed/unsigned arithmetic.

__global__ void int_promote(unsigned int *out, int *signed_in,
                             unsigned int *unsigned_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   s = signed_in[tid];
        unsigned int u = unsigned_in[tid];
        // In C, int + uint → uint (signed is converted to unsigned)
        unsigned int r = (unsigned int)(s + (int)u);
        // Bitwise: signed << with unsigned shift amount
        unsigned int shifted = (unsigned int)s << (u & 31u);
        out[tid] = r + shifted;
    }
}

// ------------------------------------------------------------------
// Multi-level pointer indirection (int**).

__global__ void double_ptr(int *out, int **pp, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = pp[tid];
        out[tid] = p[0] + p[1];
    }
}

// ------------------------------------------------------------------
// Recursive-style countdown (device helper iterating via loop).

__device__ int sum_to(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) {
        s += i;
    }
    return s;
}

__global__ void triangular(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n && data[tid] >= 0 && data[tid] <= 100) {
        out[tid] = sum_to(data[tid]);
    }
}
