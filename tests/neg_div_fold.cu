// Regression: constant-folded integer division/modulo must use C truncated
// semantics (toward zero), not Python floor semantics.
//
// C:      -7 / 2 = -3     -7 % 2 = -1
// Python: -7 // 2 = -4    -7 % 2 =  1  (floor — wrong for C!)
//
// The compiler must emit -3 / -1, not -4 / 1, in the folded constants.
__global__ void neg_div_fold(int *out) {
    // Both operands are compile-time constants — must constant-fold.
    int q = -7 / 2;   // C: -3, Python floor: -4
    int r = -7 % 2;   // C: -1, Python floor:  1
    out[0] = q;
    out[1] = r;
}
