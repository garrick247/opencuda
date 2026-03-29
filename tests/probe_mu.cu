// Probe: Struct returns with three or more early-exit paths
// - safe_sqrt: 3 return paths (negative, zero, normal)
// - categorize: 3 range-based returns
// - stats reduce: struct with multi-return combine
// - matrix power with early-exit identity

struct Result { float val; int ok; };
struct Matrix2x2 { float m00, m01, m10, m11; };

// safe_sqrt: multiple early returns before main computation
__device__ Result safe_sqrt(float x) {
    if (x < 0.0f) {
        Result r; r.val = 0.0f; r.ok = 0;
        return r;
    }
    if (x == 0.0f) {
        Result r; r.val = 0.0f; r.ok = 1;
        return r;
    }
    float g = x * 0.5f;
    for (int i = 0; i < 5; i++) {
        g = 0.5f * (g + x / g);
    }
    Result r; r.val = g; r.ok = 1;
    return r;
}

__global__ void safe_sqrt_kernel(float *out_val, int *out_ok, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Result r = safe_sqrt(in[tid]);
        out_val[tid] = r.val;
        out_ok[tid]  = r.ok;
    }
}

// Three range-based return paths
__device__ Result categorize(float x, float lo, float hi) {
    if (x < lo) {
        Result r; r.val = lo; r.ok = -1;
        return r;
    }
    if (x > hi) {
        Result r; r.val = hi; r.ok = 1;
        return r;
    }
    Result r; r.val = x; r.ok = 0;
    return r;
}

__global__ void categorize_kernel(float *out_val, int *out_ok,
                                  float *in, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Result r = categorize(in[tid], lo, hi);
        out_val[tid] = r.val;
        out_ok[tid]  = r.ok;
    }
}

// Matrix power with early-exit identity
__device__ Matrix2x2 mat_mul(Matrix2x2 a, Matrix2x2 b) {
    Matrix2x2 r;
    r.m00 = a.m00*b.m00 + a.m01*b.m10;
    r.m01 = a.m00*b.m01 + a.m01*b.m11;
    r.m10 = a.m10*b.m00 + a.m11*b.m10;
    r.m11 = a.m10*b.m01 + a.m11*b.m11;
    return r;
}

__device__ Matrix2x2 mat_pow_safe(Matrix2x2 m, int n) {
    if (n <= 0) {
        Matrix2x2 id;
        id.m00=1.0f; id.m01=0.0f; id.m10=0.0f; id.m11=1.0f;
        return id;
    }
    Matrix2x2 result;
    result.m00=1.0f; result.m01=0.0f; result.m10=0.0f; result.m11=1.0f;
    for (int i = 0; i < n; i++) {
        result = mat_mul(result, m);
    }
    return result;
}

__global__ void matrix_power_safe(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Matrix2x2 m;
        m.m00=in[0]; m.m01=in[1]; m.m10=in[2]; m.m11=in[3];
        Matrix2x2 r = mat_pow_safe(m, n);
        out[0]=r.m00; out[1]=r.m01; out[2]=r.m10; out[3]=r.m11;
    }
}

// Struct returned from multi-return fn used in loop accumulation
__global__ void sqrt_accum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum_val = 0.0f;
        int sum_ok = 0;
        for (int i = 0; i < n; i++) {
            Result r = safe_sqrt(in[i]);
            if (r.ok) {
                sum_val += r.val;
                sum_ok++;
            }
        }
        out[0] = sum_val;
        out[1] = (float)sum_ok;
    }
}
