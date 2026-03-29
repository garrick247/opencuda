// Probe: Patterns that might expose issues in the verifier
// - Phi node requires: all predecessors must provide value
// - Value used before assigned (should be caught by verifier or parser)
// - Unreachable code after return
// - Block with multiple predecessors but no phi

__global__ void diamond_cfg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        // Diamond: two paths that merge
        if (v > 0) {
            r = v * 2;
        } else {
            r = -v;
        }
        // After merge: r has value from either branch
        out[tid] = r + 1;
    }
}

// Multiple merges
__global__ void multi_diamond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a, b;
        if (v > 100) {
            a = v - 100;
        } else {
            a = 100 - v;
        }
        if (v > 50) {
            b = v - 50;
        } else {
            b = 50 - v;
        }
        out[tid] = a + b;
    }
}

// Nested diamonds
__global__ void nested_diamond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 0) {
            int inner;
            if (v > 10) {
                inner = v * 3;
            } else {
                inner = v + 10;
            }
            r = inner;
        } else {
            r = 0;
        }
        out[tid] = r;
    }
}
