// Probe: integer literal suffixes, hex/octal literals,
// preprocessor defines with expressions, float literal suffixes,
// array subscript on kernel pointer parameter

// Integer literal suffixes
__global__ void literal_suffixes(unsigned int *out) {
    unsigned int a = 42u;
    unsigned int b = 42U;
    long c = 42L;
    unsigned long d = 42UL;
    unsigned int e = 0xFF;       // hex = 255
    unsigned int f = 0xDEAD;     // hex = 57005
    int g = 0777;                // octal = 511
    out[0] = a + b;             // 84
    out[1] = (unsigned int)c + (unsigned int)d;  // 84
    out[2] = e;                 // 255
    out[3] = f;                 // 57005
    out[4] = (unsigned int)g;   // 511
}

// Float literal suffixes
__global__ void float_suffixes(float *out) {
    float a = 1.5f;
    float b = 2.5F;
    float c = 1.0f + 2.0f;   // 3.0
    float d = 0.5f * 4.0f;   // 2.0
    out[0] = a + b;   // 4.0
    out[1] = c;       // 3.0
    out[2] = d;       // 2.0
}

// Preprocessor defines with expressions
#define BLOCK_SIZE 256
#define HALF_BLOCK (BLOCK_SIZE / 2)
#define SCALE_FACTOR 3
#define SCALED_HALF (HALF_BLOCK * SCALE_FACTOR)

__global__ void preproc_expr(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid < BLOCK_SIZE) {
        int v = in[tid];
        int scaled = v * SCALE_FACTOR;
        int offset = tid - HALF_BLOCK;   // tid - 128
        out[tid] = scaled + offset;      // v*3 + tid - 128
    }
    if (tid == 0) {
        out[BLOCK_SIZE] = SCALED_HALF;   // 128*3 = 384
    }
}

// Array subscript on pointer parameters
__global__ void array_ops(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Element-wise operations using array indexing syntax
        out[tid] = a[tid] + b[tid];
        out[tid + n] = a[tid] * b[tid];
        out[tid + 2*n] = a[tid] - b[tid];
    }
}

// Negative hex and large literal
__global__ void large_literals(int *out) {
    int a = 0x7FFFFFFF;   // INT_MAX = 2147483647
    int b = -1;
    unsigned int c = 0xFFFFFFFF;  // UINT_MAX = 4294967295
    out[0] = a;    // 2147483647
    out[1] = b;    // -1
    out[2] = (int)c;  // -1 (same bit pattern)
}
