// Probe: Unusual patterns that might trip up the codegen
// - Very long integer constant (0x7FFFFFFFFFFFFFFF)
// - Floating point special values: 1.0/0.0 (inf), 0.0/0.0 (nan) at parse time
// - Negative hex constant: -0x80
// - Multiple casts in sequence

__global__ void special_consts(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long max_ll = 0x7FFFFFFFFFFFFFFFLL;
        long long min_ll = -0x7FFFFFFFFFFFFFFFLL - 1LL;
        unsigned long long max_ull = 0xFFFFFFFFFFFFFFFFULL;
        out[tid] = (max_ll >> 32) + (min_ll >> 32) + (long long)(max_ull >> 32);
    }
}

__global__ void cast_chain(float *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double d = in[tid];
        long long ll = (long long)d;
        int i = (int)ll;
        float f = (float)i;
        double d2 = (double)f;
        out[tid] = (float)d2;
    }
}

// Negative constant in various positions
__global__ void negative_consts(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = -1;
        int b = -0x80;
        int c = -255;
        int d = -(tid + 1);
        out[tid] = a + b + c + d;
    }
}
