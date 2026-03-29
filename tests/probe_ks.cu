// Probe: float constant folding (1.5f + 2.5f),
// large shift fold (1 << 30 within range),
// LICM should not hoist stores (only pure computations),
// redundant computation across blocks (CSE per-block safety),
// constant propagation through assignments

// Float constant arithmetic — should fold at compile time
__global__ void float_const_fold(float *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float a = 1.5f + 2.5f;    // should fold to 4.0f
        float b = 3.0f * 2.0f;    // should fold to 6.0f
        float c = a + b;           // should fold to 10.0f
        out[0] = c;
    }
}

// Large shift: 1 << 30 = 1073741824 (valid), 1 << 31 = -2147483648 (UB in C but common)
__global__ void large_shift_fold(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 1 << 30;   // = 0x40000000 = 1073741824
        int b = 1 << 28;   // = 0x10000000 = 268435456
        out[0] = a;
        out[1] = b;
        out[2] = a + b;    // = 0x50000000 = 1342177280
    }
}

// LICM should hoist loop-invariant COMPUTATION but NOT stores
__global__ void licm_no_hoist_store(int *out, int *in, int scale, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int inv = scale * scale;   // invariant computation — hoistable
        for (int i = 0; i < n; i++) {
            out[i] = in[i] * inv;  // store — NOT hoistable
        }
    }
}

// Chain of constant assignments: all should propagate/fold
__global__ void const_chain_prop(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 3;
        int b = a * 2;    // = 6
        int c = b + a;    // = 9
        int d = c - b;    // = 3
        int e = d * d;    // = 9
        out[0] = e;
    }
}

// Boolean int operations: (x > 0) + (x > 10) + (x > 100) - three bool flags
__global__ void bool_accumulate(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Count how many thresholds are exceeded
        int score = (v > 0) + (v > 10) + (v > 100) + (v > 1000);
        out[tid] = score;
    }
}
