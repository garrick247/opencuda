// Regression: CmpInst comparison type must follow C integer promotion.
// - INT32 lhs vs UINT32 rhs → setp.lt.u32 (unsigned wins)
// - Const lhs (INT32 default) vs UINT32 rhs → setp.lt.u32
// - UINT32 lhs vs INT32 rhs → setp.lt.u32
//
// Without this fix:
//   int n = 100; unsigned int tid = 200;
//   setp.lt.s32 → -1 < 100 == true (WRONG when tid wraps to look negative)
__global__ void cmp_type_test(int *out, int n, unsigned int limit) {
    unsigned int tid = threadIdx.x;
    // UINT32 vs INT32: must use u32 comparison
    if (tid < (unsigned int)n) {
        // UINT32 lhs, UINT32 rhs: correct as u32
        if (tid < limit) {
            out[tid] = 1;
        } else {
            out[tid] = 0;
        }
    }
    // Compare against unsigned literal — must use setp.ne.u32
    unsigned int mask = 0xFFFFFFFFu;
    if (mask != 0u) {
        out[0] = 2;
    }
}
