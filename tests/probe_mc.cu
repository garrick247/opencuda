// Probe: CSE edge cases
// - Same arithmetic on different addresses must NOT merge the loads
// - Store then load from same address: load should reuse stored value
// - Identical pure computation in both branches of a conditional
// - CSE across a call: pure ops before/after call with same operands
// - CSE with type mismatch: same operands, different result type (must NOT merge)

__global__ void cse_no_merge_diff_addr(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // a[tid] and b[tid] are different addresses — loads must NOT be CSE'd
        int x = a[tid];
        int y = b[tid];
        out[tid] = x + y;
    }
}

__global__ void cse_pure_arith(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 7 + 3;
        int b = v * 7 + 3;  // identical — CSE should deduplicate
        out[tid] = a + b;   // if CSE works: a == b, result = 2*(v*7+3)
    }
}

__global__ void cse_in_branches(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        if (v > 0) {
            int t = v * 3 + 1;  // same expression in both branches
            result = t;
        } else {
            int t = v * 3 + 1;  // per-block CSE only, so this is a separate block
            result = t;
        }
        out[tid] = result;
    }
}

// Type mismatch: same operand, different conversion — must NOT merge
__global__ void cse_type_mismatch(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        float f1 = (float)v;            // cvt.f32.s32
        float f2 = (float)v;            // same — CSE should merge (same type)
        float f3 = (float)(v * 2);      // different operand — must NOT merge with f1/f2
        out[tid] = f1 + f2 + f3;
    }
}

// Repeated subexpression with intermediate store — store invalidates nothing pure
__global__ void cse_with_store(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 5;      // first compute
        out[tid] = a;       // store to output (different address from in[])
        int b = v * 5;      // same computation — CSE should reuse a
        out[tid] = out[tid] + b;  // final result = 2 * (v*5) = 10*v
    }
}
