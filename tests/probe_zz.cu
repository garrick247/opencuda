// Probe: THE LAST PROBE — every remaining edge case in one file.
// If this compiles, we've covered everything meaningful.

// ------------------------------------------------------------------
// Recursive with float return and float params.

__device__ float lerp_recursive(float a, float b, float t, int depth) {
    if (depth <= 0) return a + (b - a) * t;
    float mid = (a + b) * 0.5f;
    if (t < 0.5f) return lerp_recursive(a, mid, t * 2.0f, depth - 1);
    return lerp_recursive(mid, b, (t - 0.5f) * 2.0f, depth - 1);
}

__global__ void lerp_rec_kernel(float *out, float *a, float *b,
                                   float *t, int depth, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = lerp_recursive(a[tid], b[tid], t[tid], depth);
}

// ------------------------------------------------------------------
// Prefix sum using ONLY warp shuffles (no shared memory).

__global__ void warp_only_scan(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = (gid < n) ? in[gid] : 0;
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if (lane >= d) v += t;
    }
    if (gid < n) out[gid] = v;
}

// ------------------------------------------------------------------
// Complex type-casting chain: half → float → int → double → long long.

__global__ void cast_chain(long long *out, unsigned short *h_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __ushort_as_half(h_in[tid]);
        float f = __half2float(h);
        int i = (int)f;
        double d = (double)i + 0.5;
        long long ll = (long long)d;
        out[tid] = ll;
    }
}

// ------------------------------------------------------------------
// Max register pressure: 30+ live variables.

__global__ void max_pressure(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float v = in[gid];
    float a0 = v + 0.1f, a1 = v + 0.2f, a2 = v + 0.3f, a3 = v + 0.4f;
    float a4 = v + 0.5f, a5 = v * 1.1f, a6 = v * 1.2f, a7 = v * 1.3f;
    float b0 = a0*a4, b1 = a1*a5, b2 = a2*a6, b3 = a3*a7;
    float c0 = b0+b1, c1 = b2+b3, c2 = a0+a7, c3 = a1+a6;
    float d0 = c0*c1, d1 = c2*c3, d2 = c0+c2, d3 = c1+c3;
    float e0 = d0+d1+d2+d3;
    float e1 = a0+a1+a2+a3+a4+a5+a6+a7;
    float e2 = b0+b1+b2+b3;
    float e3 = c0+c1+c2+c3;
    out[gid] = e0 + e1 + e2 + e3;
}

// ------------------------------------------------------------------
// Bit manipulation: reverse bits using __brev then extract fields.

__global__ void bit_reverse_extract(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned v = in[tid];
        unsigned rev = __brev(v);
        // Extract bits [7:0] from reversed
        unsigned byte0 = rev & 0xFF;
        // Extract bits [15:8]
        unsigned byte1 = (rev >> 8) & 0xFF;
        // Pack differently
        out[tid] = (byte1 << 8) | byte0;
    }
}

// ------------------------------------------------------------------
// Nested struct with 3 levels and mixed types.

struct Inner { float x; int flag; };
struct Middle { struct Inner a; struct Inner b; };
struct Outer { struct Middle m; float scale; };

__device__ float outer_compute(struct Outer o) {
    float sum = o.m.a.x + o.m.b.x;
    int flags = o.m.a.flag + o.m.b.flag;
    return sum * o.scale * (float)flags;
}

__global__ void nested3_kernel(float *out, float *ax, int *af,
                                  float *bx, int *bf,
                                  float *scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Outer o;
        o.m.a.x = ax[tid]; o.m.a.flag = af[tid];
        o.m.b.x = bx[tid]; o.m.b.flag = bf[tid];
        o.scale = scale[tid];
        out[tid] = outer_compute(o);
    }
}

// ------------------------------------------------------------------
// Final: every atomic operation in one kernel.

__global__ void all_atomics(int *i_add, int *i_min, int *i_max,
                               int *i_and, int *i_or, int *i_xor,
                               int *i_exch, int *i_cas,
                               unsigned *u_inc, unsigned *u_dec,
                               float *f_add, double *d_add,
                               int *in, float *in_f, double *in_d, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    atomicAdd(i_add, in[gid]);
    atomicMin(i_min, in[gid]);
    atomicMax(i_max, in[gid]);
    atomicAnd(i_and, in[gid]);
    atomicOr(i_or, in[gid]);
    atomicXor(i_xor, in[gid]);
    atomicExch(i_exch, in[gid]);
    atomicCAS(i_cas, 0, in[gid]);
    atomicInc(u_inc, 1000u);
    atomicDec(u_dec, 1000u);
    atomicAdd(f_add, in_f[gid]);
    atomicAdd(d_add, in_d[gid]);
}
