// Regression: ternary with two pointer array reads where CSE may not merge
// the byte-offset computations, producing duplicate cvt.s64.s32 widening.
// Without fix: three separate scale Values each trigger their own cvt.u64.u32,
// adding extra %rd registers beyond the 3 pointer params.
// With fix: _widen_by_src deduplicates within a block → only 1 cvt.u64.u32.
__global__ void ternary_dual_ptr(float *a, float *b, float *c, int n) {
    int i = threadIdx.x;
    a[i] = (i > n / 2) ? b[i] : c[i];
}
