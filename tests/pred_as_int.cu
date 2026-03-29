// Regression: predicate (bool) values used as integers.
// PTX predicate registers (%p0, %p1, ...) cannot be used directly as
// operands to add/mul/cvt/st — must use selp.type dest, 1, 0, %pred first.
// Without fix: "Arguments mismatch for instruction 'st/add/cvt'"
__global__ void pred_as_int_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int b = (v > 0);          // CmpInst → pred → stored as s32 via selp
        int c = b + 1;            // pred → int in arithmetic
        int d = (v > 0) * 5;      // pred → int → mul
        out[tid * 3 + 0] = b;
        out[tid * 3 + 1] = c;
        out[tid * 3 + 2] = d;
    }
}

__global__ void pred_to_float_test(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        float f = (float)(v != 0);  // pred → float via selp + cvt
        out[tid] = f;
    }
}
