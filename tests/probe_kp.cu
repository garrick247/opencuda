// Probe: function-like macros (with args),
// do-while(0) macro wrapper pattern,
// stringified constants via #define,
// macro that expands to a block

#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define SQUARE(x) ((x) * (x))
#define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))
#define SWAP(a, b, tmp) do { (tmp) = (a); (a) = (b); (b) = (tmp); } while(0)

// Function-like macros in kernel
__global__ void macro_arithmetic(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int clamped = CLAMP(v, -100, 100);
        int sq = SQUARE(clamped);
        out[tid] = sq;
    }
}

// MAX3 macro: nested ternary
__global__ void max3_macro(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = MAX3(a[tid], b[tid], c[tid]);
    }
}

// SWAP macro: do-while(0) wrapper with three variables
__global__ void swap_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = in[0];
        int y = in[1];
        int tmp;
        SWAP(x, y, tmp);   // expands to do { tmp=x; x=y; y=tmp; } while(0)
        out[0] = x;        // should be in[1]
        out[1] = y;        // should be in[0]
    }
}

// Macro with side-effect-free expansion used multiple times
__global__ void multi_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // SQUARE used twice: each expansion is independent
        int s1 = SQUARE(v);
        int s2 = SQUARE(v + 1);
        out[tid * 2]     = s1;
        out[tid * 2 + 1] = s2;
    }
}
