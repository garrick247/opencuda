// Probe: Tricky parsing cases around parentheses
// - Cast followed by unary minus: -(int)x
// - Cast followed by dereference: *(int*)ptr
// - Nested casts: (float)(int)(double)x
// - Cast in function argument: func((int)x, (float)y)
// - Parenthesized assignment: out[tid] = (tmp = a + b, tmp * 2)
// - sizeof expressions

__device__ int negate_cast(float x) {
    return -(int)x;
}

__device__ float cast_chain(double x) {
    return (float)(int)(x + 0.5);
}

__global__ void cast_patterns(float *out, float *in, double *din, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        double dv = din[tid];
        int ni = negate_cast(v);
        float fc = cast_chain(dv);
        int sz_int = (int)sizeof(int);
        int sz_float = (int)sizeof(float);
        out[tid] = (float)ni + fc + (float)(sz_int + sz_float);
    }
}
