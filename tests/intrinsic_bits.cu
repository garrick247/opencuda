// Bit-manipulation intrinsics: __popc, __clz, __brev
// __activemask() returns active lane bitmask (activemask.b32)
// __threadfence() emits membar.gl
// Without fix: these fall through to unknown-function path (wrong type / no emission)
__global__ void intrinsic_bits_test(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        __threadfence();
        unsigned int mask = __activemask();
        int pc = __popc(v);
        int lz = __clz(v);
        unsigned int br = __brev(v);
        out[tid * 4 + 0] = (unsigned int)pc;
        out[tid * 4 + 1] = (unsigned int)lz;
        out[tid * 4 + 2] = br;
        out[tid * 4 + 3] = mask;
    }
}
