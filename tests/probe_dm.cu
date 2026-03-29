// Probe: Patterns at the edge of SSA correctness
// - Variable written in if-body, not in else, used after merge
// - Variable written in loop init only, read after loop
// - Two variables that swap values inside a loop
// - Loop with "early continue" path that skips update

__global__ void if_write_no_else(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int modified = 0;  // default: not modified
        if (v > 0) {
            modified = v * 2;  // only written in one branch
        }
        // modified = v*2 if v>0, else 0
        out[tid] = modified + tid;
    }
}

// Value swap loop (tests that SSA correctly tracks both vars)
__global__ void swap_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = tid;
        int b = n - tid;
        for (int i = 0; i < 4; i++) {
            int tmp = a;
            a = b;
            b = tmp + 1;
        }
        out[tid] = a + b;
    }
}

// Early continue in loop
__global__ void early_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] < 0) continue;  // skip negative values
            sum += in[i];
        }
        out[tid] = sum;
    }
}
