// Probe: multiple address-of calls in same scope,
// pass-by-pointer to two different device functions in sequence,
// address-of struct field,
// pass pointer to non-local (global) array element

// Two pass-by-pointer calls in sequence — second sees first's modification
__device__ void increment(int *p) {
    *p = *p + 1;
}

__global__ void double_increment(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        increment(&v);    // v = in[tid] + 1
        increment(&v);    // v = in[tid] + 2
        out[tid] = v;
    }
}

// Pass-by-pointer to two different functions
__device__ void negate(int *p) {
    *p = -*p;
}

__global__ void inc_then_negate(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        increment(&v);    // v += 1
        negate(&v);       // v = -(v+1)
        out[tid] = v;
    }
}

// Multiple local vars, some passed by pointer, some not
__global__ void mixed_pass(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = in[tid] * 2;   // NOT passed by pointer
        increment(&a);          // a becomes in[tid]+1
        int c = a + b;         // should use updated a
        out[tid] = c;          // (in[tid]+1) + (in[tid]*2) = 3*in[tid]+1
    }
}

// Swap two variables using pass-by-pointer (like std::swap)
__device__ void swap_ints(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void swap_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        int x = in[tid * 2];
        int y = in[tid * 2 + 1];
        // Swap if x > y to get sorted pair
        if (x > y) {
            swap_ints(&x, &y);
        }
        out[tid * 2]     = x;
        out[tid * 2 + 1] = y;
    }
}

// Pass pointer to two locals in same call (both modified)
__device__ void add_and_sub(int *a, int *b, int delta) {
    *a = *a + delta;
    *b = *b - delta;
}

__global__ void balance_op(int *out, int *in, int n, int delta) {
    int tid = threadIdx.x;
    if (tid < n) {
        int p = in[tid * 2];
        int q = in[tid * 2 + 1];
        add_and_sub(&p, &q, delta);
        out[tid * 2]     = p;
        out[tid * 2 + 1] = q;
    }
}
