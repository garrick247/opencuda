// Regression: constant-folded overflow must wrap to type's bit width.
// Without wrapping:
//   0xFFFFFFFFu + 1u → _fold_bin returns 4294967296 (Python int, no wrap)
//   Const(UINT32, 4294967296) → str(4294967296) → ptxas rejects u32 immediate
__global__ void uint_overflow_test(unsigned int *out) {
    // 0xFFFFFFFF + 1 wraps to 0 for u32
    unsigned int x = 0xFFFFFFFFu + 1u;
    // INT32 overflow: INT32_MAX + 1 wraps to INT32_MIN = -2147483648
    int y = 2147483647 + 1;
    out[0] = x;
    out[1] = (unsigned int)y;
}
