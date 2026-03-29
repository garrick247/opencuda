// Regression: atomicSub must SUBTRACT, not add.
// PTX has no atom.sub; correct implementation is atom.add(-val).
// Without this fix: atomicSub(ptr, 5) emits atom.add(ptr, 5) — adds instead!
__global__ void atomic_sub_test(int *out, int *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        // Decrement out[0] by in[i] — must use atom.global.add with negated value
        atomicSub(out, in[i]);
    }
}

// Constant version: atomicSub with a literal constant.
// Should emit atom.global.add.s32 dest, [ptr], -5  (NOT +5).
__global__ void atomic_sub_const(int *out) {
    atomicSub(out, 5);
}
