// Probe: type punning, union-like reinterpret patterns, mixed-precision
// accumulation, and multi-level pointer out-params.

// ------------------------------------------------------------------
// Float bits reinterpret via pointer cast (common GPU trick).

__global__ void float_bits(unsigned int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        unsigned int bits = *((unsigned int *)&v);
        out[tid] = bits;
    }
}

// ------------------------------------------------------------------
// Reverse: bits back to float.

__global__ void bits_float(float *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int b = in[tid];
        float v = *((float *)&b);
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Mixed int/float accumulation in a loop.

__global__ void mixed_accum(float *out, int *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            int raw = in[tid * k + i];
            acc += (float)raw * 0.001f;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Integer-promoted boolean in accumulation.

__global__ void bool_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        for (int i = 0; i < 8; i++) {
            count += (v >> i) & 1;  // popcount via bool accumulation
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Chained pointer out-params (result built through multiple helpers).

__device__ void split_int(int v, int *hi, int *lo) {
    *hi = v >> 16;
    *lo = v & 0xFFFF;
}

__device__ void combine_int(int hi, int lo, int *out) {
    *out = (hi << 16) | (lo & 0xFFFF);
}

__global__ void split_combine(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int hi, lo;
        split_int(in[tid], &hi, &lo);
        int r;
        combine_int(lo, hi, &r);  // swap high/low halves
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Double→float→int narrowing chain with clamp.

__global__ void narrow_chain(int *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double d = in[tid];
        float f = (float)d;
        if (f > 32767.0f) f = 32767.0f;
        if (f < -32768.0f) f = -32768.0f;
        out[tid] = (int)f;
    }
}

// ------------------------------------------------------------------
// Long long bit manipulation.

__global__ void ll_bits(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        long long hi = (v >> 32) & 0xFFFFFFFFL;
        long long lo = v & 0xFFFFFFFFL;
        out[tid] = (lo << 32) | hi;  // swap halves
    }
}

// ------------------------------------------------------------------
// Unsigned overflow as wraparound sentinel.

__global__ void wrap_sentinel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int result = v;
        for (int i = 0; i < 8; i++) {
            result *= result;  // intentional overflow
        }
        out[tid] = (int)(result & 0xFF);
    }
}

// ------------------------------------------------------------------
// Ternary inside array index.

__global__ void ternary_index(int *out, int *a, int *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = sel[tid];
        int idx = (s > 0) ? tid : (n - 1 - tid);
        out[tid] = a[idx] + b[idx];
    }
}
