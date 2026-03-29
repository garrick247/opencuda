// Regression: expf/logf must use base-2 scaling, NOT direct ex2/lg2.
// ex2.approx.f32 computes 2^x; expf(x) needs 2^(x*log2e).
// lg2.approx.f32 computes log2(x); logf(x) needs lg2(x)*ln(2).
__global__ void math_explog_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val = in[tid];
        out[tid * 4 + 0] = expf(val);       // 2^(val*log2e)
        out[tid * 4 + 1] = logf(val);       // lg2(val)*ln2
        out[tid * 4 + 2] = exp2f(val);      // 2^val (direct)
        out[tid * 4 + 3] = log2f(val);      // log2(val) (direct)
    }
}
