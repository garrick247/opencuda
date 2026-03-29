// Probe: long type, size_t, long long/int mixing, deeply nested ternary,
// ternary with function calls on branches, and complex macro arguments.

// ------------------------------------------------------------------
// long type: on GPU, long is 32-bit (like int in most ABI).

__global__ void long_arithmetic(long *out, long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long v = in[tid];
        long r = v * 3L + 7L;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// size_t arithmetic (equivalent to u64 on GPU).

__global__ void sizet_arith(size_t *out, size_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        size_t v = in[tid];
        out[tid] = v + sizeof(double);  // v + 8
    }
}

// ------------------------------------------------------------------
// Mixed int / long long arithmetic.

__global__ void int_ll_mix(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        long long big = (long long)v * (long long)v;  // v^2 as 64-bit
        long long r   = big + (long long)v;           // v^2 + v
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Deeply nested ternary (5 levels).

__global__ void nested_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = (v < 0)   ? -100 :
                (v < 10)  ? 0    :
                (v < 50)  ? 1    :
                (v < 100) ? 2    :
                            3;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Ternary with device function calls on branches.

__device__ int slow_path(int v) { return v * v - v; }
__device__ int fast_path(int v) { return v + v; }

__global__ void ternary_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = (v > 0) ? fast_path(v) : slow_path(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Macro with complex argument: array subscript and pointer deref.

#define DOUBLE_VAL(x) ((x) + (x))

__global__ void macro_complex_arg(int *out, int *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Complex argument: array subscript and pointer arithmetic
        int i = idx[tid] % n;
        int r = DOUBLE_VAL(in[i]);   // expands to (in[i]) + (in[i])
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Unsigned long long arithmetic.

__global__ void ull_arith(unsigned long long *out,
                           unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        unsigned long long r = v * 0xDEADBEEFULL + 1ULL;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// long long comparison and conditional.

__global__ void ll_compare(int *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        long long threshold = 1000000000LL;   // 1 billion
        out[tid] = (v > threshold) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Shift by long long amount (cast to int for shift).

__global__ void ll_shift(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        int shift = tid % 63;
        long long r = v << shift;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// ptrdiff_t-like: pointer subtraction (yields ptrdiff_t = s64 on GPU).

__global__ void ptr_diff(long long *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // pointer subtraction gives element count
        long long diff = (long long)(b - a);
        out[tid] = diff + (long long)tid;
    }
}
