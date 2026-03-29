// Regression: 'int x = uint_expr' must insert cvt.s32.u32 so that the variable
// carries the declared signed type. Without the fix, the variable stays UINT32,
// and pointer arithmetic uses cvt.u64.u32 (zero-extend) instead of cvt.s64.s32
// (sign-extend), causing wrong addresses for negative indices.
//
// Key patterns:
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;  // UINT32 expr → must coerce to INT32
//   int offset = tid - radius;                         // result now INT32, not UINT32
//   in[tid - 1] offset widened with cvt.s64.s32 ✓
__global__ void int_decl_uint_assign(float *out, float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0 && tid < n) {
        // tid must be INT32 so tid-1 is INT32 and cvt.s64.s32 is used
        int prev = tid - 1;
        out[tid] = in[tid] - in[prev];
    }
}
