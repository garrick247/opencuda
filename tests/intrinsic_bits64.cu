// Regression: 64-bit variants of bit intrinsics.
// __popcll(ull) must emit popc.b64, not popc.b32 (source is 64-bit).
// __clzll(ll)  must emit clz.b64, not clz.b32.
// __brevll(ull) must emit brev.b64 with a 64-bit dest register.
__global__ void intrinsic_bits64_test(unsigned long long *out,
                                       unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        int pc   = __popcll(v);          // popc.b64 — source is 64-bit
        int lz   = __clzll((long long)v); // clz.b64  — source is 64-bit
        unsigned long long br = __brevll(v); // brev.b64 — src and dest are 64-bit
        out[tid * 3 + 0] = (unsigned long long)pc;
        out[tid * 3 + 1] = (unsigned long long)lz;
        out[tid * 3 + 2] = br;
    }
}
