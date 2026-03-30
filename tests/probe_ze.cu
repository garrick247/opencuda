// Probe: final coverage sweep — rsqrtf, __fdividef, __powf, __sinf/__cosf,
// __expf/__logf/__log2f/__log10f, __tanf, fast math intrinsics,
// __uint2float_rn/__float2uint_rz, __double2uint_rn/__uint2double_rn,
// and __sincosf if available.

// ------------------------------------------------------------------
// rsqrtf (fast reciprocal square root).

__global__ void rsqrtf_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = rsqrtf(in[tid]);
}

// ------------------------------------------------------------------
// __fdividef (fast float division).

__global__ void fdividef_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __fdividef(a[tid], b[tid]);
}

// ------------------------------------------------------------------
// __powf (fast power).

__global__ void powf_kernel(float *out, float *base, float *exp_v, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __powf(base[tid], exp_v[tid]);
}

// ------------------------------------------------------------------
// __sinf / __cosf (fast trig).

__global__ void fast_trig(float *out_s, float *out_c, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_s[tid] = __sinf(in[tid]);
        out_c[tid] = __cosf(in[tid]);
    }
}

// ------------------------------------------------------------------
// __expf / __logf / __log2f / __log10f (fast math).

__global__ void fast_exp_log(float *out_e, float *out_l,
                                float *out_l2, float *out_l10,
                                float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_e[tid]   = __expf(in[tid]);
        out_l[tid]   = __logf(in[tid]);
        out_l2[tid]  = __log2f(in[tid]);
        out_l10[tid] = __log10f(in[tid]);
    }
}

// ------------------------------------------------------------------
// __tanf (fast tangent).

__global__ void fast_tan(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = __tanf(in[tid]);
}

// ------------------------------------------------------------------
// __uint2float_rn / __float2uint_rz.

__global__ void uint_float_cvt(float *out_f, unsigned *out_u,
                                   unsigned *in_u, float *in_f, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_f[tid] = __uint2float_rn(in_u[tid]);
        out_u[tid] = __float2uint_rz(in_f[tid]);
    }
}

// ------------------------------------------------------------------
// __double2uint_rn / __uint2double_rn.

__global__ void double_uint_cvt(double *out_d, unsigned *out_u,
                                    unsigned *in_u, double *in_d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_d[tid] = __uint2double_rn(in_u[tid]);
        out_u[tid] = __double2uint_rn(in_d[tid]);
    }
}

// ------------------------------------------------------------------
// Combined fast math expression: sigmoid using __expf.

__global__ void fast_sigmoid(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = in[gid];
        out[gid] = 1.0f / (1.0f + __expf(-v));
    }
}

// ------------------------------------------------------------------
// Fast swish: x * sigmoid(x) using fast math.

__global__ void fast_swish(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = in[gid];
        float sig = 1.0f / (1.0f + __expf(-v));
        out[gid] = v * sig;
    }
}
