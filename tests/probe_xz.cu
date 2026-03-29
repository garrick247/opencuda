// Probe: inline PTX asm (asm volatile), __builtin_expect, restrict + aliasing,
// static __device__ function, multi-level nested struct, function with 8+ params,
// ternary with side effects in loop, __float2half_rn (default rounding),
// __half2float, cascaded type casts, and unsigned char arithmetic saturation.

// ------------------------------------------------------------------
// Inline PTX asm: simple register move.

__global__ void asm_mov(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        asm volatile("mov.s32 %0, %1;" : "=r"(r) : "r"(v));
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// __builtin_expect (hint to compiler; should parse and ignore).

__global__ void builtin_expect_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (__builtin_expect(tid < n, 1)) {
        out[tid] = in[tid] * 2;
    }
}

// ------------------------------------------------------------------
// static __device__ function (file-local linkage).

static __device__ int clamp_s32(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

__global__ void clamp_kernel(int *out, int *in, int lo, int hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = clamp_s32(in[tid], lo, hi);
}

// ------------------------------------------------------------------
// Multi-level nested struct.

struct Vec2 { float x, y; };
struct Vec4 { struct Vec2 lo, hi; };

__device__ float vec4_sum(struct Vec4 v) {
    return v.lo.x + v.lo.y + v.hi.x + v.hi.y;
}

__global__ void nested_struct_kernel(float *out, float *ax, float *ay,
                                       float *bx, float *by,
                                       float *cx, float *cy,
                                       float *dx, float *dy, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec4 v;
        v.lo.x = ax[tid]; v.lo.y = ay[tid];
        v.hi.x = bx[tid]; v.hi.y = by[tid];
        out[tid] = vec4_sum(v);
    }
}

// ------------------------------------------------------------------
// Function with 10 scalar parameters.

__device__ int poly10(int a0, int a1, int a2, int a3, int a4,
                       int a5, int a6, int a7, int a8, int a9) {
    return a0 + a1*2 + a2*3 + a3*4 + a4*5 +
           a5*6 + a6*7 + a7*8 + a8*9 + a9*10;
}

__global__ void poly10_kernel(int *out, int *a, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = a[tid];
        out[tid] = poly10(v, v+1, v+2, v+3, v+4,
                          v+5, v+6, v+7, v+8, v+9);
    }
}

// ------------------------------------------------------------------
// Ternary with side effects in a loop.

__global__ void ternary_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0, prev = 0;
        for (int i = 0; i < 16; i++) {
            int v = in[(tid + i) % n];
            s += (v > prev) ? v : -v;
            prev = v;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// __float2half_rn (default rounding half conversion).

__global__ void float2half_rn(unsigned short *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __float2half_rn(in[tid]);
        out[tid] = __half_as_ushort(h);
    }
}

// ------------------------------------------------------------------
// __half2float (exact, no rounding).

__global__ void half2float_kernel(float *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __ushort_as_half(in[tid]);
        out[tid] = __half2float(h);
    }
}

// ------------------------------------------------------------------
// Cascaded casts: int → float → double → int.

__global__ void cascade_cast(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   i = in[tid];
        float f = (float)i;
        double d = (double)f + 0.5;
        out[tid] = (int)d;
    }
}

// ------------------------------------------------------------------
// unsigned char arithmetic (8-bit wrapping).

__global__ void uchar_arith(unsigned char *out, unsigned char *a,
                               unsigned char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char sum = a[tid] + b[tid];  // wraps mod 256
        unsigned char diff = a[tid] - b[tid];
        out[tid * 2]     = sum;
        out[tid * 2 + 1] = diff;
    }
}

// ------------------------------------------------------------------
// __ldg variants: __ldg on int, float, unsigned.

__global__ void ldg_types(int *out_i, float *out_f, unsigned *out_u,
                            int *in_i, float *in_f, unsigned *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_i[tid] = __ldg(in_i + tid);
        out_f[tid] = __ldg(in_f + tid);
        out_u[tid] = __ldg(in_u + tid);
    }
}
