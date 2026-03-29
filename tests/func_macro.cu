// Function-like macro expansion regression test.
// Without fix: #define MAX(a,b) not expanded — 'MAX' parsed as unknown identifier.
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(v, lo, hi) (MIN(MAX((v), (lo)), (hi)))
#define SQ(x) ((x) * (x))

__global__ void func_macro_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid * 4 + 0] = MAX(v, 0.0f);
        out[tid * 4 + 1] = MIN(v, 1.0f);
        out[tid * 4 + 2] = CLAMP(v, 0.0f, 1.0f);
        out[tid * 4 + 3] = SQ(v);
    }
}
