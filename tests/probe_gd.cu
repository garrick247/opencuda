// Probe: __device__ function that modifies a parameter via pointer,
// multi-level indirection (*ptr = val, **ptr = val),
// function taking struct by pointer and returning by value

struct Matrix2x2 {
    float a, b, c, d;
};

__device__ float det2(Matrix2x2 m) {
    return m.a * m.d - m.b * m.c;
}

__device__ Matrix2x2 inv2(Matrix2x2 m) {
    float d = det2(m);
    Matrix2x2 r;
    if (d == 0.0f) {
        r.a = 0.0f; r.b = 0.0f;
        r.c = 0.0f; r.d = 0.0f;
    } else {
        float inv_d = 1.0f / d;
        r.a =  m.d * inv_d;
        r.b = -m.b * inv_d;
        r.c = -m.c * inv_d;
        r.d =  m.a * inv_d;
    }
    return r;
}

__device__ Matrix2x2 matmul2(Matrix2x2 a, Matrix2x2 b) {
    Matrix2x2 r;
    r.a = a.a * b.a + a.b * b.c;
    r.b = a.a * b.b + a.b * b.d;
    r.c = a.c * b.a + a.d * b.c;
    r.d = a.c * b.b + a.d * b.d;
    return r;
}

__global__ void mat2_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 4;
        Matrix2x2 m;
        m.a = in[base]; m.b = in[base+1];
        m.c = in[base+2]; m.d = in[base+3];
        Matrix2x2 inv = inv2(m);
        Matrix2x2 prod = matmul2(m, inv);
        // prod should be identity (approximately)
        out[tid] = prod.a + prod.d - 2.0f;  // should be ~0
    }
}
