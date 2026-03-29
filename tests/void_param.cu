// Regression: (void) parameter list in __device__ functions
// Without fix: ParseError "expected IDENT, got RPAREN ')'"
// Fix: _parse_device_func and _parse_kernel skip 'void' before ')' as empty param list

__device__ int get_magic(void) {
    return 42;
}

__device__ float get_pi(void) {
    return 3.14159f;
}

__global__ void void_param_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = get_magic() + (int)get_pi();
    }
}
