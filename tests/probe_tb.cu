// Probe: high register pressure (many live vars), multiple parallel
// accumulators, deep struct field chains, and __noinline__ device fns.

struct Pair { float lo, hi; };
struct Quad { float a, b, c, d; };

// ------------------------------------------------------------------
// Many live variables: 8 parallel accumulators.

__global__ void eight_accum(float *out, float *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (int i = 0; i < k; i++) {
            float v = in[tid * k + i];
            a0 += v;
            a1 += v * v;
            a2 += v * v * v;
            a3 += v * (float)i;
            a4 += (v > 0.0f) ? v : 0.0f;
            a5 += (v < 0.0f) ? -v : 0.0f;
            a6 += (v > 1.0f) ? 1.0f : 0.0f;
            a7 += (v > 0.0f && v < 1.0f) ? v : 0.0f;
        }
        out[tid * 8 + 0] = a0;
        out[tid * 8 + 1] = a1;
        out[tid * 8 + 2] = a2;
        out[tid * 8 + 3] = a3;
        out[tid * 8 + 4] = a4;
        out[tid * 8 + 5] = a5;
        out[tid * 8 + 6] = a6;
        out[tid * 8 + 7] = a7;
    }
}

// ------------------------------------------------------------------
// __noinline__ device function.

__device__ __noinline__ float expensive(float v, int iters) {
    float acc = v;
    for (int i = 0; i < iters; i++) {
        acc = acc * 0.999f + (float)i * 0.001f;
    }
    return acc;
}

__global__ void noinline_call(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = expensive(in[tid], 10);
    }
}

// ------------------------------------------------------------------
// Struct returned from device function, fields deeply used.

__device__ Pair minmax_pair(float a, float b) {
    Pair p;
    p.lo = (a < b) ? a : b;
    p.hi = (a > b) ? a : b;
    return p;
}

__global__ void struct_return_use(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair p = minmax_pair(a[tid], b[tid]);
        out[tid * 2 + 0] = p.lo;
        out[tid * 2 + 1] = p.hi;
    }
}

// ------------------------------------------------------------------
// Quad struct: 4-field computation.

__device__ Quad quad_op(float v) {
    Quad q;
    q.a = v;
    q.b = v * v;
    q.c = v * v * v;
    q.d = v + q.a + q.b + q.c;
    return q;
}

__global__ void quad_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Quad q = quad_op(in[tid]);
        out[tid] = q.a + q.b + q.c + q.d;
    }
}

// ------------------------------------------------------------------
// Two shared memory reductions in sequence.

__global__ void dual_reduce(float *osum, float *omax, float *in, int n) {
    __shared__ float ssum[256], smax[256];
    int tid = threadIdx.x;
    ssum[tid] = (tid < n) ? in[tid] : 0.0f;
    smax[tid] = (tid < n) ? in[tid] : -1e30f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            smax[tid] = (smax[tid] > smax[tid + s]) ? smax[tid] : smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(osum, ssum[0]);
        // no atomicMax for float — just store (single block assumed)
        *omax = smax[0];
    }
}

// ------------------------------------------------------------------
// Interleaved int and float registers.

__global__ void interleaved_types(int *iout, float *fout, int *iin, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int iv = iin[tid];
        float fv = fin[tid];
        int ia = iv * 3 + 7;
        float fa = fv * 2.5f - 1.0f;
        int ib = ia ^ (ia >> 1);
        float fb = fa * fa + fa;
        iout[tid] = ib;
        fout[tid] = fb;
    }
}

// ------------------------------------------------------------------
// Deeply nested struct field access.

struct Inner { int x, y; };
struct Outer { struct Inner p, q; int tag; };

__global__ void nested_struct(int *out, struct Outer *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Outer o = in[tid];
        int r = o.p.x + o.p.y + o.q.x + o.q.y + o.tag;
        out[tid] = r;
    }
}
