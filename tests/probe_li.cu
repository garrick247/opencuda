// Probe: integer overflow/wraparound in constant expressions,
// CSE across multiple instructions in same block,
// LICM: loop-invariant expression hoisted out of loop,
// device function with early return via if-else (no mid-if return),
// large constant expressions

// Integer constant folding: overflow behavior
__global__ void const_overflow(int *out) {
    // 2147483647 + 1 wraps to -2147483648 (C undefined, but compiler should not crash)
    int a = 2147483647;
    int b = a + 1;   // wraparound in 32-bit
    out[0] = b;
    // Shift by 31: still defined
    int c = 1 << 31;   // -2147483648 (assuming two's complement)
    out[1] = c;
}

// CSE: common subexpression across multiple uses in same block
__global__ void cse_multi_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // The expression (v * 3 + 7) computed twice — CSE should merge
        int a = (v * 3 + 7) * 2;
        int b = (v * 3 + 7) + 10;
        out[tid * 2]     = a;
        out[tid * 2 + 1] = b;
    }
}

// LICM: loop-invariant computation
__global__ void licm_invariant(int *out, int *in, int n, int scale) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int factor = scale * scale;   // loop-invariant (doesn't change in loop)
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i] * factor;   // factor hoistable
        }
        out[0] = sum;
    }
}

// Device function with if-else return (both branches return, no mid-if return)
__device__ int safe_div(int a, int b) {
    if (b == 0) {
        return 0;
    } else {
        return a / b;
    }
}

__global__ void call_safe_div(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_div(in[tid * 2], in[tid * 2 + 1]);
    }
}

// Large constant chain: multiple const operations
__global__ void const_chain(int *out) {
    int a = 1000000;
    int b = a * 1000;       // 1e9
    int c = b / 1000;       // back to 1e6
    int d = c + 500000;     // 1.5e6
    int e = d * 2 - 1000000; // 2e6
    out[0] = e;   // 2000000
}
