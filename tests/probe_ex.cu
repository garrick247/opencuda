// Probe: function with many parameters (>8), parameter list wrapping
// Also: functions called with many arguments, complex argument expressions

__device__ float poly10(float x, float c0, float c1, float c2, float c3,
                         float c4, float c5, float c6, float c7, float c8,
                         float c9) {
    return c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 +
           x * (c5 + x * (c6 + x * (c7 + x * (c8 + x * c9))))))));
}

__global__ void eval_poly(float *out, float *x_vals, float *coeffs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = x_vals[tid];
        float r = poly10(x,
                         coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                         coeffs[4], coeffs[5], coeffs[6], coeffs[7],
                         coeffs[8], coeffs[9]);
        out[tid] = r;
    }
}

// Mixed complex argument expressions
__device__ int multi_arg(int a, int b, int c, int d, int e,
                          int f, int g, int h) {
    return ((a + b) * (c - d)) ^ ((e | f) & (g + h));
}

__global__ void complex_args(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = multi_arg(v, v+1, v*2, v/2, v&3,
                             v|4, v^5, v%7 + 1);
    }
}
