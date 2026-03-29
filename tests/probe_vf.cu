// Probe: 64-bit arithmetic edge cases, fused multiply-add, rsqrt,
// const pointer parameters, and mixed 32/64 operations.

// ------------------------------------------------------------------
// 64-bit arithmetic: add, mul, shl with long long.

__global__ void ll_arith(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        long long a = v + 1LL;
        long long b = v * 2LL;
        long long c = v >> 1;
        long long d = v << 3;
        long long e = a + b + c + d;
        out[tid] = e;
    }
}

// ------------------------------------------------------------------
// Mixed 32/64 bit: widening multiply result.

__global__ void wide_mul(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // 32×32 → 64 widening
        long long r = (long long)v * (long long)(v + 1);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// rsqrtf and fused multiply-add.

__global__ void fast_math_ops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float pos_v = v > 0.0f ? v : -v;  // |v|
        // rsqrtf: 1/sqrt(v) — fast approximation
        float r = rsqrtf(pos_v + 1.0f);
        // fmaf: fused multiply-add (a*b + c, single rounding)
        float fma_r = fmaf(v, v, r);  // v^2 + rsqrt(|v|+1)
        out[tid] = fma_r;
    }
}

// ------------------------------------------------------------------
// const pointer parameter — should work like regular pointer for reads.

__global__ void const_ptr_read(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];          // read through const ptr
        int w = in[tid + (tid < n - 1 ? 1 : 0)];  // read neighbor
        out[tid] = v + w;
    }
}

// ------------------------------------------------------------------
// Array parameter syntax (decays to pointer in C).

__device__ int sum4(int arr[4]) {
    return arr[0] + arr[1] + arr[2] + arr[3];
}

__global__ void array_param_decay(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int local[4];
        local[0] = in[tid];
        local[1] = in[tid] + 1;
        local[2] = in[tid] + 2;
        local[3] = in[tid] + 3;
        out[tid] = sum4(local);
    }
}

// ------------------------------------------------------------------
// Double FMA.

__global__ void double_fma(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        // fma(a, b, c) = a*b + c in double precision
        double r = fma(v, v, v);  // v^2 + v
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// 64-bit comparison and branch.

__global__ void ll_compare(int *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        int r = 0;
        if (v > 0LL)          r += 1;
        if (v < -1000000LL)   r += 2;
        if (v == 42LL)        r += 4;
        if (v != 0LL)         r += 8;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Unsigned 64-bit.

__global__ void ull_ops(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        unsigned long long a = v >> 32;   // high 32 bits
        unsigned long long b = v & 0xFFFFFFFFULL;  // low 32 bits
        unsigned long long c = (a << 32) | b;      // reassemble
        out[tid] = c;
    }
}
