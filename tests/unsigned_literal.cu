// Regression: unsigned integer literals must be typed UINT32, not INT32.
// - Literals with u/U suffix: 5u, 0xFFFFFFFFu → UINT32
// - Large hex literals with no suffix: 0xFFFFFFFF > INT32_MAX → UINT32
//
// Without this fix:
//   0xFFFFFFFF → Const(INT32, 4294967295) — value doesn't fit in s32!
//   Ballot mask 0xFFFFFFFF would use s32 where u32 is required.
__global__ void unsigned_literal_test(unsigned int *out, int n) {
    unsigned int tid = threadIdx.x;
    // 0xFFFFFFFF as ballot-style full-warp mask — must be UINT32
    unsigned int full_mask = 0xFFFFFFFF;
    // Explicit u suffix
    unsigned int x = 5u;
    // Compare against large unsigned constant — must use setp.lt.u32
    if (tid < (unsigned int)n) {
        out[tid] = full_mask & x;
    }
}
