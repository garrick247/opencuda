// Probe: char/byte type correctness across operations —
// signed char vs unsigned char comparison semantics,
// char arithmetic promotion to int,
// byte array stride with pointer arithmetic,
// char cast chain

// Signed char: values -128 to 127; setp should use s8→s32 comparison
__global__ void signed_char_ops(char *out, char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        char v = in[tid];
        // Signed char range check: -100 to 100
        char clamped = (v < (char)-100) ? (char)-100 : ((v > (char)100) ? (char)100 : v);
        out[tid] = clamped;
    }
}

// Unsigned char: 0-255, arithmetic wraps modulo 256
__global__ void uchar_arith(unsigned char *out, unsigned char *a,
                              unsigned char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char x = a[tid];
        unsigned char y = b[tid];
        unsigned char sum = x + y;    // truncated to 8 bits
        unsigned char diff = x - y;   // wraps modulo 256
        out[tid * 2]     = sum;
        out[tid * 2 + 1] = diff;
    }
}

// Char to int widening and back
__global__ void char_to_int(int *out, char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        char c = in[tid];
        int i = (int)c;           // sign-extend to 32-bit
        unsigned int u = (unsigned int)(unsigned char)c;  // zero-extend
        out[tid * 2]     = i;
        out[tid * 2 + 1] = (int)u;
    }
}

// Struct with char fields: correct byte offsets
typedef struct {
    char r, g, b, a;   // 4 bytes total (1 each)
} ByteColor;

__global__ void byte_color_ops(int *out, ByteColor *colors, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        ByteColor c = colors[tid];
        // Sum the 4 channels (widened to int)
        int sum = (int)c.r + (int)c.g + (int)c.b + (int)c.a;
        out[tid] = sum;
    }
}

// Char pointer arithmetic: step by 2 bytes (every other byte)
__global__ void char_stride2(unsigned char *out, unsigned char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid * 2];    // stride 2 bytes
    }
}
