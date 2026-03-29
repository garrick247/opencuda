// Probe: double-precision intrinsics (__dadd_rn, __dmul_rn, __ddiv_rn, __drcp_rn,
// __dsqrt_rn, __dfma_rn), int32↔float promotion in mixed arithmetic,
// unsigned char array stride patterns, uint8 bit manipulation,
// reading past struct end via pointer (field_ptr+1), ternary with
// mismatched types (int vs float → float promotion), and
// negative constant expressions in initializers.

// ------------------------------------------------------------------
// Double-precision intrinsics.

__global__ void dadd_dmul(double *out_a, double *out_m,
                            double *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_a[tid] = __dadd_rn(a[tid], b[tid]);
        out_m[tid] = __dmul_rn(a[tid], b[tid]);
    }
}

__global__ void ddiv_drcp(double *out_d, double *out_r, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_d[tid] = __ddiv_rn(1.0, in[tid]);
        out_r[tid] = __drcp_rn(in[tid]);
    }
}

__global__ void dsqrt_dfma(double *out_s, double *out_f,
                              double *a, double *b, double *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_s[tid] = __dsqrt_rn(a[tid]);
        out_f[tid] = __dfma_rn(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// int32 ↔ float promotion in mixed arithmetic expressions.

__global__ void mixed_int_float(float *out, int *ia, float *fb, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   i = ia[tid];
        float f = fb[tid];
        // int + float → float
        float r1 = i + f;
        // int * float → float
        float r2 = i * f;
        // (int / int) = truncated int, then + float
        float r3 = (i / 3) + f;
        out[tid] = r1 + r2 + r3;
    }
}

// ------------------------------------------------------------------
// uint8 bit manipulation via shifts and masks.

__global__ void uint8_bits(unsigned char *out, unsigned char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char v = in[tid];
        // Reverse nibbles
        unsigned char lo = v & 0x0F;
        unsigned char hi = (v >> 4) & 0x0F;
        out[tid] = (unsigned char)((lo << 4) | hi);
    }
}

// ------------------------------------------------------------------
// Ternary with mismatched int/float types (int promoted to float).

__global__ void ternary_type_promo(float *out, int *cond, int *iv, float *fv, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // ternary: int vs float → both become float
        float r = (cond[tid] > 0) ? (float)iv[tid] : fv[tid];
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Negative constant in array initializer and subtraction.

__global__ void neg_const_init(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lut[4] = {-10, -5, 5, 10};
        int idx = tid & 3;
        out[tid] = lut[idx] * 2 + (-3);  // -3 as a negative constant
    }
}

// ------------------------------------------------------------------
// Unsigned char stride: accessing every 3rd byte in a byte array.

__global__ void uchar_stride3(int *out, unsigned char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Access bytes at offset tid*3, tid*3+1, tid*3+2
        unsigned char b0 = in[tid * 3    ];
        unsigned char b1 = in[tid * 3 + 1];
        unsigned char b2 = in[tid * 3 + 2];
        // Pack into 24-bit value
        out[tid] = ((int)b0 << 16) | ((int)b1 << 8) | (int)b2;
    }
}

// ------------------------------------------------------------------
// Double accumulator with mixed int loop variable.

__global__ void double_int_mix(double *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            // int / int → int (truncated), then converted to double for add
            s += (double)(in[i]) / (double)(i + 1);
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// __dsqrt_rn on small/large values.

__global__ void dsqrt_range(double *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = (double)(tid + 1);
        out[tid] = __dsqrt_rn(v * v + 1.0);  // sqrt(n^2 + 1) ≈ n for large n
    }
}

// ------------------------------------------------------------------
// Int to double to int round-trip.

__global__ void int_double_trip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   x = in[tid];
        double d = (double)x + 0.5;  // add 0.5
        int   r = (int)d;             // truncate back (should be x or x+1 if x>=0)
        out[tid] = r - x;             // should be 0 or 1 depending on original
    }
}
