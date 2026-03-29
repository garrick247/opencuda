// Probe: complex type cast chains, implicit conversions in expressions,
// signed/unsigned overflow behavior, and mixed-width arithmetic.

// ------------------------------------------------------------------
// Float to int conversion rounding modes.

__global__ void float_to_int_modes(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid * 4 + 0] = (int)v;            // truncate
        out[tid * 4 + 1] = __float2int_rn(v); // round nearest
        out[tid * 4 + 2] = __float2int_rd(v); // round down (floor)
        out[tid * 4 + 3] = __float2int_ru(v); // round up (ceil)
    }
}

// ------------------------------------------------------------------
// Int to float rounding.

__global__ void int_to_float(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = __int2float_rn(v);
    }
}

// ------------------------------------------------------------------
// Chained widening: int8 -> int16 -> int32 -> int64 -> float -> double.

__global__ void chain_widen(double *out, signed char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed char b  = in[tid];
        short        s  = (short)b;
        int          i  = (int)s;
        long long    ll = (long long)i;
        float        f  = (float)ll;
        double       d  = (double)f;
        out[tid] = d;
    }
}

// ------------------------------------------------------------------
// Narrowing chain: double -> float -> int -> short -> char.

__global__ void chain_narrow(signed char *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double  d = in[tid];
        float   f = (float)d;
        int     i = (int)f;
        short   s = (short)i;
        out[tid]  = (signed char)s;
    }
}

// ------------------------------------------------------------------
// Unsigned overflow arithmetic (wraps at 2^32).

__global__ void uint_overflow(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // These should all wrap without UB (unsigned overflow is defined in C)
        out[tid * 3 + 0] = v + 0xFFFFFFFFu;  // v - 1 (wraps)
        out[tid * 3 + 1] = v * 0xFFFFu;       // deliberate large multiply
        out[tid * 3 + 2] = ~v;                 // bitwise NOT
    }
}

// ------------------------------------------------------------------
// Mixed short/int arithmetic.

__global__ void short_int_mix(int *out, short *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = (int)a[tid];  // widen explicitly
        int bv = b[tid];
        out[tid] = av * bv + av - bv;
    }
}

// ------------------------------------------------------------------
// Pointer cast for reinterpretation (int → float bits).

__global__ void bits_reinterpret(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = __int_as_float(v);
    }
}

// ------------------------------------------------------------------
// Float bits as int and back.

__global__ void float_int_roundtrip(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int bits = __float_as_int(v);
        // Flip sign bit
        bits ^= 0x80000000;
        out[tid] = __int_as_float(bits);
    }
}

// ------------------------------------------------------------------
// Long-long to double and back.

__global__ void ll_double_roundtrip(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        double d = __ll2double_rn(v);
        out[tid] = __double2ll_rn(d);
    }
}
