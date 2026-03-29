// Regression: __half / __half2 type keywords not recognized
// Without fix: lexer had 'half' → KW_HALF but not '__half' or '__half2' →
//   ParseError "expected type, got '__half'".
// Fix: added '__half' and '__half2' → KW_HALF to lexer keyword table.

__global__ void half_param_test(__half *out, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fa = __half2float(a[tid]);
        float fb = __half2float(b[tid]);
        out[tid] = __float2half(fa + fb);
    }
}

// __half as local variable type
__global__ void half_local(__half *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __float2half(in[tid]);
        float f = __half2float(h);
        out[tid] = __float2half(f * 2.0f);
    }
}
