// Probe: nested device calls as arguments (f(g(x))),
// device function used in array index expression,
// three levels of device function inlining,
// device function with multiple paths that all return

// Level 1: simple transform
__device__ int double_it(int x) {
    return x * 2;
}

// Level 2: calls double_it
__device__ int quad_it(int x) {
    return double_it(double_it(x));  // 4x
}

// Level 3: calls quad_it
__device__ int octo_it(int x) {
    return quad_it(double_it(x));  // 8x
}

// Kernel using 3-level chain
__global__ void three_level_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = octo_it(in[tid]);   // should be in[tid] * 8
    }
}

// Device function used in two-arg call to another device fn
__device__ int add_vals(int a, int b) {
    return a + b;
}

__device__ int abs_neg(int x) {
    return x < 0 ? -x : x;
}

__global__ void nested_as_args(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // add_vals(abs_neg(a[tid]), abs_neg(b[tid]))
        int result = add_vals(abs_neg(a[tid]), abs_neg(b[tid]));
        out[tid] = result;
    }
}

// Device function with early return from if-branch
// (known limitation: only simple single-return path supported)
__device__ int safe_recip(int x) {
    if (x == 0) return 0;
    return 1000 / x;
}

__global__ void use_safe_recip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_recip(in[tid]);
    }
}

// Multiple same device function calls on different values
__global__ void multi_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = double_it(in[tid]);
        int b = double_it(in[tid] + 1);
        int c = double_it(in[tid] - 1);
        out[tid * 3]     = a;
        out[tid * 3 + 1] = b;
        out[tid * 3 + 2] = c;
    }
}
