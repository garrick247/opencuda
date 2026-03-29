// Probe: __sad/__usad, __double2int_rn/__int2double_rn, __umul64hi,
// atomic disambiguation (unsigned vs float), integer promotion subtleties,
// and compound bitwise patterns.

// ------------------------------------------------------------------
// __sad: sum of absolute differences (a - b, take abs, add c).

__global__ void sad_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = __sad(a[tid], b[tid], 0);   // |a - b|
        out[tid] = r;
    }
}

__global__ void usad_kernel(unsigned int *out,
                             unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int r = __usad(a[tid], b[tid], 0u);   // |a - b| (unsigned)
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// __double2int_rn / __int2double_rn explicit type conversions.

__global__ void double_int_cvt(int *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double d = in[tid];
        int i = __double2int_rn(d);     // round to nearest
        double back = __int2double_rn(i);
        // back ≈ round(d), store difference from original
        out[tid] = i;
    }
}

// ------------------------------------------------------------------
// __float2int_rz / __int2float_rn.

__global__ void float_int_cvt(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = in[tid];
        int i   = __float2int_rz(f);   // truncate toward zero
        float b = __int2float_rn(i);   // back to float
        out[tid] = (int)b;             // same as i
    }
}

// ------------------------------------------------------------------
// __float_as_int / __int_as_float: bit reinterpretation.

__global__ void float_bits_reinterpret(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = in[tid];
        int bits = __float_as_int(f);
        float back = __int_as_float(bits);
        out[tid] = (back == f) ? 1 : 0;   // should always be 1
    }
}

// ------------------------------------------------------------------
// __double_as_longlong / __longlong_as_double.

__global__ void double_bits_reinterpret(long long *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double d = in[tid];
        long long bits = __double_as_longlong(d);
        double back = __longlong_as_double(bits);
        out[tid] = (back == d) ? 1LL : 0LL;  // should always be 1
    }
}

// ------------------------------------------------------------------
// __mulhi: high 32 bits of 32x32 multiply.

__global__ void mulhi_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // __mulhi(a, b) = upper 32 bits of a*b (signed)
        int hi = __mulhi(a[tid], b[tid]);
        out[tid] = hi;
    }
}

// ------------------------------------------------------------------
// __umulhi: high 32 bits of unsigned 32x32 multiply.

__global__ void umulhi_kernel(unsigned int *out,
                               unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int hi = __umulhi(a[tid], b[tid]);
        out[tid] = hi;
    }
}

// ------------------------------------------------------------------
// __ffs: find first set bit (1-indexed from LSB).

__global__ void ffs_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = __ffs(v);   // 1-indexed position of LSB, 0 if v==0
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Integer promotion in mixed-type arithmetic.

__global__ void int_promotion(int *out, short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        short s = in[tid];
        int r = s * 1000;      // short promoted to int before multiply
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Bitwise NOT and complement patterns.

__global__ void bitwise_not(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int r = ~v;          // bitwise complement
        unsigned int s = ~v & 0xFF;   // mask lower byte of complement
        out[tid] = r ^ s;             // XOR
    }
}
