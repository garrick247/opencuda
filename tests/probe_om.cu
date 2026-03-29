// Probe: chained casts, address-of global var, complex initializers,
// conditional expression as function argument.

// ------------------------------------------------------------------
// Chained casts: (double)(float)(int)x — must chain conversions correctly.
// (double)(float) should round through float precision, not convert directly.

__global__ void chained_cast(double *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // Chain: int -> float -> double
        double r = (double)(float)v;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Cast from unsigned char: (int)(unsigned char) must zero-extend, not sign-extend.
// Data byte at index i; treating as u8 and widening to s32.

__global__ void uchar_widen(int *out, unsigned char *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // unsigned char with high bit set: 0xFF -> 255, not -1
        int v = (int)data[tid];
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Address-of global device variable passed to a device function.
// Tests that &g_var compiles to a GlobalAddrInst and is passed correctly.

__device__ int g_shared_val;

__device__ void increment_global(int *ptr, int delta) {
    *ptr += delta;
}

__global__ void addr_of_global(int delta) {
    int tid = threadIdx.x;
    if (tid == 0) {
        increment_global(&g_shared_val, delta);
    }
}

// ------------------------------------------------------------------
// Conditional expression as function call argument.
// f(a > 0 ? a : -a) — ternary as argument.

__device__ float safe_sqrt(float x) {
    return x >= 0.0f ? __fsqrt_rn(x) : 0.0f;
}

__global__ void ternary_as_arg(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        // ABS via ternary, then sqrt
        out[tid] = safe_sqrt(v >= 0.0f ? v : -v);
    }
}

// ------------------------------------------------------------------
// Negative array index (with guard): data[tid - offset] where offset > 0.
// When tid >= offset, this is safe. Tests that cvt.s64.s32 is used for the
// signed subtraction result (not cvt.u64.u32 which would wrap negative).

__global__ void neg_offset(float *out, float *data, int offset, int n) {
    int tid = threadIdx.x;
    if (tid >= offset && tid < n) {
        out[tid] = data[tid - offset];
    }
}
