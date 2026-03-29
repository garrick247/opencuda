// Regression: address-of struct field: &b.member where b is a local struct
// Without fix: ParseError "unexpected token ';'" on the next statement
// Fix: generic &expr fallback in _parse_unary_expr spills scalar to .local
//      and returns a pointer (same pattern as &scalar_local for plain vars)

struct Vec4 {
    float x, y, z, w;
};

__global__ void field_ptr(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec4 v;
        v.x = in[tid];
        v.y = in[tid] * 2.0f;
        v.z = in[tid] * 3.0f;
        v.w = in[tid] * 4.0f;
        // Take address of field and write through pointer
        float *px = &v.x;
        *px = in[tid] + 1.0f;
        out[tid] = v.y + v.z + v.w;
    }
}

struct Pair {
    int first;
    int second;
};

__global__ void int_field_ptr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair p;
        p.first = in[tid];
        p.second = in[tid] * 2;
        int *pf = &p.first;
        *pf = in[tid] - 1;
        out[tid] = p.second;
    }
}
