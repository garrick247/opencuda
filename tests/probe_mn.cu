// Probe: Struct compound assignment on fields, chained inline calls,
//        and immediate field access on inline return value
// - s.x += expr (compound assign on struct field)
// - f(g(a, b)).field (inline result immediately field-accessed)
// - Two back-to-back inline calls returning same struct type
// - Struct with conditional field init (one field set only in if-branch)

struct V2 { float x, y; };
struct Mat2 { float a, b, c, d; };  // 2x2 matrix row-major

__device__ V2 v2_add(V2 a, V2 b) {
    V2 r; r.x = a.x + b.x; r.y = a.y + b.y; return r;
}

__device__ V2 v2_scale(V2 v, float s) {
    V2 r; r.x = v.x * s; r.y = v.y * s; return r;
}

__device__ V2 mat2_mul(Mat2 m, V2 v) {
    V2 r;
    r.x = m.a * v.x + m.b * v.y;
    r.y = m.c * v.x + m.d * v.y;
    return r;
}

// Compound assign on struct field: acc.x += ...; acc.y += ...;
__global__ void transform_sum(float *out, float *vecs, float *mat, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Mat2 m; m.a=mat[0]; m.b=mat[1]; m.c=mat[2]; m.d=mat[3];
        V2 acc; acc.x = 0.0f; acc.y = 0.0f;
        for (int i = 0; i < n; i++) {
            V2 v; v.x = vecs[i*2]; v.y = vecs[i*2+1];
            V2 tv = mat2_mul(m, v);
            acc.x += tv.x;
            acc.y += tv.y;
        }
        out[0] = acc.x; out[1] = acc.y;
    }
}

// Back-to-back inline calls returning same struct type
__global__ void add_then_scale(float *out, float *a, float *b, float s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        V2 va; va.x = a[tid*2]; va.y = a[tid*2+1];
        V2 vb; vb.x = b[tid*2]; vb.y = b[tid*2+1];
        V2 sum = v2_add(va, vb);
        V2 res = v2_scale(sum, s);
        out[tid*2] = res.x; out[tid*2+1] = res.y;
    }
}

// Struct with conditional field override
__device__ V2 conditional_v2(float x, float y, int flip) {
    V2 r; r.x = x; r.y = y;
    if (flip) { r.x = -x; r.y = -y; }
    return r;
}

__global__ void conditional_transform(float *out, float *in, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        V2 v = conditional_v2(in[tid*2], in[tid*2+1], flags[tid]);
        V2 s = v2_add(v, v);
        out[tid*2] = s.x; out[tid*2+1] = s.y;
    }
}
