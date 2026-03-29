// Probe: __any_sync / __all_sync / __ballot_sync return value usage,
// predicate from comparison used directly in expressions,
// warp-level vote in complex control flow

__global__ void ballot_compress(int *out, int *count, int *in, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int active = (tid < n) ? 1 : 0;
    unsigned int ballot = __ballot_sync(0xffffffff, active);
    // Count set bits in ballot
    int cnt = __popc(ballot);
    if (lane == 0) {
        out[tid >> 5] = cnt;
    }
    if (tid == 0) *count = cnt;
}

// __any_sync in loop exit condition
__global__ void any_done(int *out, int *work, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? work[tid] : 0;
    int done = 0;
    for (int iter = 0; iter < 100 && !done; iter++) {
        val = val * 2 + 1;
        int my_done = (val > 1000) ? 1 : 0;
        done = __any_sync(0xffffffff, my_done);
    }
    if (tid < n) out[tid] = val;
}

// __all_sync for convergence check
__global__ void all_converged(float *out, float *in, float tol, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    int converged = 0;
    for (int i = 0; i < 50 && !converged; i++) {
        float new_val = val * 0.9f + 0.1f;
        float diff = new_val - val;
        if (diff < 0.0f) diff = -diff;
        val = new_val;
        int my_ok = (diff < tol) ? 1 : 0;
        converged = __all_sync(0xffffffff, my_ok);
    }
    if (tid < n) out[tid] = val;
}
