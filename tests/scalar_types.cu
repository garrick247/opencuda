// Regression: scalar type coverage — bool, fixed-width int types,
// char, short, size_t, and widening initializer conversions.
// Without fixes:
//   bool pos = v > 0;        → ParseError "undefined variable 'bool'"
//   int32_t x = ...;         → ParseError "undefined variable 'int32_t'"
//   size_t n;                → ParseError "undefined variable 'size_t'"
//   long long v = in[tid];   → no widening CvtInst → shl.b64 %r1 (s32!), ...

__global__ void bool_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        bool pos = v > 0;
        bool neg = v < 0;
        bool zero = v == 0;
        out[tid] = (int)pos - (int)neg + (int)zero;
    }
}

__global__ void fixed_width_int(int32_t *out, uint32_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint32_t v = in[tid];
        int32_t s = (int32_t)v - 100;
        int64_t wide = (int64_t)s * 1000LL;
        out[tid] = (int32_t)(wide >> 10);
    }
}

__global__ void size_t_kernel(float *out, float *in, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

__global__ void widening_init(double *out, float *in, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double fd = in[tid];        // float → double widening in decl
        double fi = iin[tid];       // int → double widening in decl
        long long ll = iin[tid];    // int → long long widening in decl
        float fll = (float)ll;
        out[tid] = fd + fi + fll;
    }
}

__global__ void char_short_test(int *out, char *ca, short *sb, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        char c = ca[tid];
        short s = sb[tid];
        unsigned char uc = (unsigned char)(c + 1);
        unsigned short us = (unsigned short)(s + 1);
        out[tid] = (int)c + (int)s + (int)uc + (int)us;
    }
}
