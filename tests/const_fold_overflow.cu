// Regression: constant folding overflow.
// 1 << 31 = 2147483648 in Python (unbounded), but s32 max is 2147483647.
// Without fix: emitter writes 'mov.s32 %r0, 2147483648' which ptxas rejects.
// With fix: result is masked to s32 range → -2147483648 (INT32_MIN).
#define SIGN_BIT (1 << 31)
#define ALL_ONES  (~0)
__global__ void const_fold_overflow_test(int *out, unsigned int *uout, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Shift overflow: 1 << 31 should fold to INT32_MIN (-2147483648)
        out[tid * 3 + 0] = SIGN_BIT;
        // Bitwise NOT of 0: ~0 should fold to -1 (s32)
        out[tid * 3 + 1] = ALL_ONES;
        // Unsigned shift: (unsigned)0xFFFFFFFF >> 1 = 0x7FFFFFFF = 2147483647
        uout[tid] = 0xFFFFFFFFu >> 1;
    }
}
