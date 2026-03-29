// Comprehensive atomic operations test covering all PTX type requirements:
// - atomicAdd: atom.add.s32 (typed)
// - atomicMin/Max: atom.min.s32/u32 (typed)
// - atomicAnd/Or/Xor: atom.and.b32/or.b32/xor.b32 (bitwise, NOT s32!)
// - atomicExch: atom.exch.b32 (bitwise)
// - atomicCAS: atom.cas.b32 dest, [addr], compare, val (3-arg!)
__global__ void atomic_ops_test(int *flags, unsigned int *ubuf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(flags, 1);
        atomicMin(flags, tid);
        atomicMax(flags, tid);
        atomicAnd(ubuf, (unsigned int)tid);
        atomicOr(ubuf, (unsigned int)tid);
        atomicXor(ubuf, (unsigned int)tid);
        atomicExch(flags, 0);
        atomicCAS(flags, 0, 1);
    }
}
