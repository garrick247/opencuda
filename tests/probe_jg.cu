// Probe: LICM correctness — loop-invariant expressions should be hoisted,
// but loop-variant ones must NOT be hoisted.
// Also test: unrolled loops with break/continue correctness

// Loop with truly invariant computation — should be hoisted
__global__ void licm_invariant(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float scale = a[0] * b[0];  // reads from array but constant indices
        float result = 0.0f;
        for (int i = 0; i < n; i++) {
            result += a[i] * scale;  // scale is loop-invariant
        }
        out[tid] = result;
    }
}

// Loop with condition that changes each iteration — must NOT be hoisted
__global__ void licm_variant_condition(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = in[i];         // variant: different each iteration
            if (v > sum) {           // condition uses sum which changes
                sum = v;
            }
        }
        *out = sum;
    }
}

// Small counted loop (trip count ≤ 16) — should be unrolled
__global__ void unroll_4x(float *out, float *in) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += in[i];
    }
    out[tid] = sum;
}

// Counted loop with conditional break — must NOT be unrolled incorrectly
__global__ void loop_with_break(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            if (in[i] < 0) break;
            sum += in[i];
        }
        *out = sum;
    }
}
