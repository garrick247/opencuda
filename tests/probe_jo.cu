// Probe: ternary operator as argument to device function call,
// chained ternary (a ? b : c ? d : e),
// device function with pointer output parameter (write-back via pointer),
// device function called in loop condition

// Device function: clamp a value to [lo, hi]
__device__ int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Ternary as argument to clamp
__global__ void ternary_as_arg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // ternary feeds directly into function call argument
        out[tid] = clamp_i(v >= 0 ? v : -v, 0, 100);
    }
}

// Chained ternary: classify into one of four ranges
__global__ void chained_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int category = v < 0   ? 0 :
                       v < 10  ? 1 :
                       v < 100 ? 2 : 3;
        out[tid] = category;
    }
}

// Device function with pointer output (write-back pattern)
__device__ void minmax(int *lo, int *hi, int a, int b) {
    *lo = a < b ? a : b;
    *hi = a < b ? b : a;
}

__global__ void call_minmax(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mn = in[0], mx = in[0];
        for (int i = 1; i < n; i++) {
            int lo, hi;
            minmax(&lo, &hi, mn, in[i]);
            mn = lo;
            mx = mx > hi ? mx : hi;
        }
        out[0] = mn;
        out[1] = mx;
    }
}

// Nested ternary used in array indexing
__global__ void ternary_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Read from either in[tid] or in[n-1-tid] based on which half
        int idx = tid < n/2 ? tid : n - 1 - tid;
        out[tid] = in[idx];
    }
}
