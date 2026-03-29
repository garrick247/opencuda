// Probe: struct with array fields, pointer-to-struct __device__ function args,
// call chains (depth 3+), and mixed struct/scalar __device__ function patterns.

// ------------------------------------------------------------------
// Struct with array field: inline array member access.

struct Poly4 {
    float c[4];  // polynomial coefficients c0..c3
};

__device__ float eval_poly4(Poly4 p, float x) {
    return p.c[0] + p.c[1]*x + p.c[2]*x*x + p.c[3]*x*x*x;
}

__global__ void poly_eval_kernel(float *out, float *coeffs, float *xs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Poly4 p;
        p.c[0] = coeffs[tid * 4 + 0];
        p.c[1] = coeffs[tid * 4 + 1];
        p.c[2] = coeffs[tid * 4 + 2];
        p.c[3] = coeffs[tid * 4 + 3];
        out[tid] = eval_poly4(p, xs[tid]);
    }
}

// ------------------------------------------------------------------
// __device__ function taking pointer-to-struct, reading via ->.

struct Vec3f {
    float x, y, z;
};

__device__ float dot3(Vec3f *a, Vec3f *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

__device__ float len3(Vec3f *v) {
    return dot3(v, v);
}

__global__ void dot_lengths(float *out_dot, float *out_len, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 6;
        Vec3f a, b;
        a.x = data[base + 0]; a.y = data[base + 1]; a.z = data[base + 2];
        b.x = data[base + 3]; b.y = data[base + 4]; b.z = data[base + 5];
        out_dot[tid] = dot3(&a, &b);
        out_len[tid]  = len3(&a);
    }
}

// ------------------------------------------------------------------
// 3-deep call chain: f3 calls f2 calls f1.

__device__ float f1(float x) { return x * 2.0f + 1.0f; }
__device__ float f2(float x) { return f1(x) * f1(x - 1.0f); }
__device__ float f3(float x) { return f2(x) + f2(x + 1.0f); }

__global__ void chain3_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = f3(in[tid]);
    }
}

// ------------------------------------------------------------------
// __device__ function: modify struct through output pointer.

struct Stats {
    float sum;
    float sum_sq;
    int   count;
};

__device__ void accumulate(Stats *s, float v) {
    s->sum    += v;
    s->sum_sq += v * v;
    s->count++;
}

__global__ void stats_kernel(float *out_sum, float *out_sq, int *out_cnt,
                              float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s;
        s.sum = 0.0f; s.sum_sq = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            accumulate(&s, data[i]);
        }
        out_sum[0] = s.sum;
        out_sq[0]  = s.sum_sq;
        out_cnt[0] = s.count;
    }
}

// ------------------------------------------------------------------
// Struct with two array fields of different types.

struct Pair4 {
    int   keys[4];
    float vals[4];
};

__global__ void pair_scatter(int *out_k, float *out_v, float *raw, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair4 p;
        p.keys[0] = (int)raw[tid * 8 + 0];
        p.keys[1] = (int)raw[tid * 8 + 1];
        p.keys[2] = (int)raw[tid * 8 + 2];
        p.keys[3] = (int)raw[tid * 8 + 3];
        p.vals[0] = raw[tid * 8 + 4];
        p.vals[1] = raw[tid * 8 + 5];
        p.vals[2] = raw[tid * 8 + 6];
        p.vals[3] = raw[tid * 8 + 7];
        int best_k = p.keys[0];
        float best_v = p.vals[0];
        for (int i = 1; i < 4; i++) {
            if (p.vals[i] > best_v) {
                best_v = p.vals[i];
                best_k = p.keys[i];
            }
        }
        out_k[tid] = best_k;
        out_v[tid] = best_v;
    }
}
