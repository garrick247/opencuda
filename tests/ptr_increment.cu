// Regression: pointer postfix ++/-- and prefix ++/-- as statements
// Without fix: ParseError "expected SEMI, got PLUSPLUS '++'"
// Fix: _parse_stmt handles PLUSPLUS/MINUSMINUS after _parse_lvalue_or_expr

__global__ void ptr_advance(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *src = in + tid;
        float *dst = out + tid;
        float val = *src;
        src++;          // pointer postfix increment
        dst++;          // pointer postfix increment
        src--;          // pointer postfix decrement
        ++src;          // pointer prefix increment (via unary)
        *dst = val;
    }
}

__global__ void int_ptr_advance(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = in + tid;
        int *q = out + tid;
        int v = *p;
        p++;
        q++;
        *q = v;
    }
}
