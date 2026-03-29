// Regression: 64-bit integer literals with LL/ULL suffixes must produce INT64/UINT64.
// Without fix: 0xFFFFFFFFFFFFFFFFULL parsed as Const(UINT32, ...) — wrong type
// causes wrong PTX register type and potentially truncated value.
__global__ void int64_literals_test(long long *out, unsigned long long *uout, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long a = 0x7FFFFFFFFFFFFFFFLL;   // INT64_MAX
        long long b = -1LL;                    // -1 as int64
        unsigned long long c = 0xFFFFFFFFFFFFFFFFULL;  // UINT64_MAX
        unsigned long long d = 1ULL << 40;    // value > UINT32_MAX
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
        uout[tid * 2 + 0] = c;
        uout[tid * 2 + 1] = d;
    }
}
