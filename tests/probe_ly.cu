// Probe: LICM edge cases
// - Loop with loop-invariant expression AND conditional break
// - Nested loops: inner-invariant hoisted to inner preheader only
// - Loop where invariant computation uses a loop-carried value (must NOT hoist)
// - Multiple invariant chains: A=f(x), B=g(A), C=h(B) — hoist in order

__global__ void licm_break(float *out, float *in, int n, float threshold) {
    int tid = threadIdx.x;
    if (tid < n) {
        float scale = threshold * 2.0f;  // loop-invariant
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += in[i] * scale;
            if (sum > 1e6f) break;
        }
        out[tid] = sum;
    }
}

__global__ void licm_nested(float *out, float *in, int n, float alpha) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = alpha * alpha;      // outer-loop invariant
        float b = a + 1.0f;           // outer-loop invariant
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            float inner_scale = b * in[i];  // inner-loop invariant (varies by i)
            float row_sum = 0.0f;
            for (int j = 0; j < n; j++) {
                row_sum += in[j] * inner_scale;
            }
            total += row_sum;
        }
        out[tid] = total;
    }
}

// Invariant that depends on loop-carried variable: must NOT be hoisted
__global__ void licm_no_hoist(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
            // sum * 2 depends on sum which is loop-carried — cannot hoist
            out[i] = sum * 2;
        }
        out[n] = sum;
    }
}

// Chain of invariants: A → B → C, all should be hoisted in order
__global__ void licm_chain(float *out, float *in, int n,
                             float x, float y, float z) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = x * y;          // loop-invariant
        float b = a + z;          // loop-invariant (depends on a)
        float c = b * b - a;      // loop-invariant (depends on b and a)
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += in[i] * c;
        }
        out[tid] = sum;
    }
}
