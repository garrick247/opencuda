// Probe: negative indices, negative constants, negation operator edge cases

__global__ void negative_const(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Negative constant in expression
        int r = v + (-1);
        int s = v * (-2);
        int t = -v + (-v);  // double negation
        out[tid] = r + s + t;
    }
}

// Unary minus on various types
__global__ void unary_minus_types(float *out_f, int *out_i, float *in_f, int *in_i, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_f[tid] = -in_f[tid];
        out_i[tid] = -in_i[tid];
    }
}

// Integer negation of unsigned (wraps)
__global__ void neg_unsigned(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid] = (unsigned int)(-(int)v);
    }
}

// Subtraction from zero (equivalent to negation)
__global__ void sub_from_zero(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = 0.0f - in[tid];
    }
}
