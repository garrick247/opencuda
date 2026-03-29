// Probe: complex do-while conditions, multiple assignments in one statement,
// integer overflow patterns, bitfield manipulation, and boundary conditions.

// ------------------------------------------------------------------
// do-while with complex condition (multiple variables updated).

__global__ void dowhile_complex(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 0xFF;
        int a = 0, b = 1, iters = 0;
        do {
            int tmp = (a + b) & 0xFF;
            a = b;
            b = tmp;
            iters++;
        } while (b != 1 && iters < 128);
        out[tid] = iters;  // Fibonacci cycle length mod 256
    }
}

// ------------------------------------------------------------------
// Chained ternary with side effects via function calls.

__device__ int expensive1(int x) { return x * x + 1; }
__device__ int expensive2(int x) { return x * 3 - 1; }
__device__ int expensive3(int x) { return (x + 7) & ~7; }

__global__ void chained_ternary_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = (v > 0) ? expensive1(v) :
                (v < 0) ? expensive2(-v) :
                          expensive3(0);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Multiple return values via output array parameter.

__device__ void compute_stats(int *data, int n, int *out_min, int *out_max,
                               int *out_sum) {
    int mn = data[0], mx = data[0], sm = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sm += data[i];
    }
    *out_min = mn;
    *out_max = mx;
    *out_sum = sm;
}

__global__ void stats_kernel(int *out, int *in, int stride, int chunk, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mn, mx, sm;
        compute_stats(in + tid * stride, chunk, &mn, &mx, &sm);
        out[tid * 3 + 0] = mn;
        out[tid * 3 + 1] = mx;
        out[tid * 3 + 2] = sm;
    }
}

// ------------------------------------------------------------------
// Integer overflow: wrap-around behavior (unsigned).

__global__ void unsigned_wrap(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int wrapped = v + 0xFFFFFFFF;  // wraps to v - 1
        out[tid] = wrapped;
    }
}

// ------------------------------------------------------------------
// Bit tricks: swap two values without a temp.

__global__ void xor_swap(int *arr, int i, int j, int n) {
    if (i < n && j < n && i != j) {
        arr[i] = arr[i] ^ arr[j];
        arr[j] = arr[i] ^ arr[j];
        arr[i] = arr[i] ^ arr[j];
    }
}

// ------------------------------------------------------------------
// Round up to next power of 2.

__device__ unsigned int next_pow2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__global__ void pow2_kernel(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = next_pow2(in[tid]);
    }
}

// ------------------------------------------------------------------
// Signed-to-unsigned boundary: abs of INT_MIN is undefined in C, but
// we test the bitwise behavior.

__global__ void signed_unsigned_edge(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Safe absolute value: uses conditional without overflow
        int absv = (v < 0) ? (int)((unsigned int)(-v)) : v;
        // For INT_MIN, -v overflows to INT_MIN; cast to unsigned stays as is
        out[tid] = absv;
    }
}

// ------------------------------------------------------------------
// Float rounding: different rounding modes via casts.

__global__ void float_rounding(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float rn = roundf(v);    // round to nearest
        float rf = floorf(v);    // round toward -inf
        float rc = ceilf(v);     // round toward +inf
        float rz = truncf(v);    // round toward zero
        out[tid] = rn + rf + rc + rz;
    }
}

// ------------------------------------------------------------------
// Multiple outputs to global memory in one kernel.

__global__ void multi_write_global(int *a, int *b, int *c, int *d,
                                    int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        a[tid] = v;
        b[tid] = v * 2;
        c[tid] = v * 3;
        d[tid] = v + 100;
    }
}

// ------------------------------------------------------------------
// Bitfield packing: store 4 nibbles in one int.

__device__ unsigned int pack_nibbles(int a, int b, int c, int d) {
    return ((unsigned int)(a & 0xF) <<  0) |
           ((unsigned int)(b & 0xF) <<  4) |
           ((unsigned int)(c & 0xF) <<  8) |
           ((unsigned int)(d & 0xF) << 12);
}

__global__ void nibble_pack(unsigned int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = pack_nibbles(a[tid], b[tid], c[tid], d[tid]);
    }
}
