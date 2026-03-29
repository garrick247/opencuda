// Probe: __int2half_rn/__half2int_rn conversions, struct with array member,
// complex for-init with multiple vars, large switch (>8 cases),
// deep nested ternary, and __ll2half_rn/__half2ll_rn.

// ------------------------------------------------------------------
// __int2half_rn: convert int → half.

__global__ void int2half_kernel(unsigned short *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __int2half_rn(in[tid]);
        out[tid] = __half_as_ushort(h);
    }
}

// ------------------------------------------------------------------
// __half2int_rn / __half2uint_rn: convert half → int.

__global__ void half2int_kernel(int *out_i, unsigned *out_u,
                                  unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __ushort_as_half(in[tid]);
        out_i[tid] = __half2int_rn(h);
        out_u[tid] = __half2uint_rn(h);
    }
}

// ------------------------------------------------------------------
// __float2half_rz / __half2float: round-toward-zero conversion.

__global__ void float_half_rz(unsigned short *out_h, float *out_f,
                                float *in_f, unsigned short *in_h, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __float2half_rz(in_f[tid]);
        float f  = __half2float(__ushort_as_half(in_h[tid]));
        out_h[tid] = __half_as_ushort(h);
        out_f[tid] = f;
    }
}

// ------------------------------------------------------------------
// Struct with fixed-size array member.

struct BufInt4 {
    int data[4];
    int len;
};

__device__ int buf_sum(struct BufInt4 b) {
    int s = 0;
    for (int i = 0; i < b.len && i < 4; i++) s += b.data[i];
    return s;
}

__global__ void buf_kernel(int *out, int *a, int *b, int *c, int *d,
                              int *lens, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct BufInt4 buf;
        buf.data[0] = a[tid];
        buf.data[1] = b[tid];
        buf.data[2] = c[tid];
        buf.data[3] = d[tid];
        buf.len = lens[tid];
        out[tid] = buf_sum(buf);
    }
}

// ------------------------------------------------------------------
// Large switch (12 cases) with explicit values.

__device__ int classify12(int v) {
    switch (v % 12) {
        case  0: return 100;
        case  1: return 101;
        case  2: return 102;
        case  3: return 103;
        case  4: return 104;
        case  5: return 105;
        case  6: return 106;
        case  7: return 107;
        case  8: return 108;
        case  9: return 109;
        case 10: return 110;
        case 11: return 111;
        default: return -1;
    }
}

__global__ void switch12_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = classify12(in[tid]);
}

// ------------------------------------------------------------------
// Complex for-init: two loop variables.

__global__ void two_var_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int i, j;
        for (i = 0, j = tid; i < j; i++, j--) {
            s += i + j;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Deep nested ternary (4 levels).

__device__ int deep_ternary(int v) {
    return v < 0   ? -2 :
           v == 0  ? 0  :
           v < 10  ? 1  :
           v < 100 ? 2  : 3;
}

__global__ void deep_ternary_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = deep_ternary(in[tid]);
}
