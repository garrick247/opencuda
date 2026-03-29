// Probe: real-world patterns from production CUDA code
// - atomicCAS loop (compare-and-swap spin loop)
// - __ballot_sync usage with bit operations
// - warp-level prefix sum
// - multiple __shared__ + __syncthreads() barriers
// - do { ... } while(0) idiom (common in macros expanded to statements)

__global__ void atomic_cas_update(int *target, int old_val, int new_val, int *success) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int prev = atomicCAS(target, old_val, new_val);
        *success = (prev == old_val) ? 1 : 0;
    }
}

__global__ void ballot_count(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
        // popcount of mask = number of lanes with v > 0
        int count = 0;
        unsigned int m = mask;
        while (m) {
            count += (int)(m & 1u);
            m >>= 1;
        }
        out[tid] = count;
    }
}

__global__ void do_while_macro_idiom(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        // Simulate "do { ... } while(0)" macro expansion
        do {
            if (v < 0) { result = -1; break; }
            if (v == 0) { result = 0; break; }
            result = 1;
        } while (0);
        out[tid] = result;
    }
}
