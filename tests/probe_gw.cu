// Probe: integer promotions in expressions — char and short arithmetic
// should promote to int; unsigned char + signed int rules

__global__ void int_promotion(int *out, char *in_c, short *in_s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        char c = in_c[tid];
        short s = in_s[tid];
        // Arithmetic promotes char/short to int
        int r = c + s;  // c and s promoted
        int r2 = c * s;
        int r3 = (int)c + (int)s;  // explicit cast — same result
        out[tid] = r + r2 + r3;
    }
}

// Mixed signed/unsigned in same expression (C integer rank rules)
__global__ void signed_unsigned_mix(int *out, int *si, unsigned int *ui, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sv = si[tid];
        unsigned int uv = ui[tid];
        // Adding signed to unsigned — result is unsigned
        unsigned int r = (unsigned int)sv + uv;
        // Comparison: signed < unsigned (C converts signed to unsigned)
        int cmp = (sv < (int)uv) ? 1 : 0;
        out[tid] = (int)r + cmp;
    }
}

// Truncation: cast larger to smaller type
__global__ void truncation(char *out_c, short *out_s,
                              int *in_i, long long *in_ll, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in_i[tid];
        long long w = in_ll[tid];
        // Truncate to smaller types
        out_c[tid] = (char)(v & 0xFF);
        out_s[tid] = (short)(w & 0xFFFF);
    }
}
