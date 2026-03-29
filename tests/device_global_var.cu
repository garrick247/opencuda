// Regression: __device__ global variables at module level.
// Without fix: parse_module always called _parse_device_func() which
// expected LPAREN → ParseError "expected LPAREN, got SEMI".
// Fix: lookahead past qualifiers — if LPAREN comes before SEMI, it's a
// function; otherwise parse as a module-level global variable.

__device__ int global_counter;
__device__ float scale_factor;
__device__ int lookup_table[16];

__global__ void increment(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = atomicAdd(&global_counter, 1);
    }
}

__global__ void scale(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * scale_factor;
    }
}

__global__ void table_lookup(int *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        if (i >= 0 && i < 16) {
            out[tid] = lookup_table[i];
        } else {
            out[tid] = -1;
        }
    }
}
