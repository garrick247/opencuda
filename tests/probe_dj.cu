// Probe: Stress test for the for-loop with complex multi-level nesting
// - 4-level nested loops
// - Loop indices used in complex array addressing
// - Accumulator updated inside deeply nested loops

__global__ void deep_nest_4(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        float total = 0.0f;
        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                for (int c = 0; c < 4; c++) {
                    for (int d = 0; d < 4; d++) {
                        int idx = (a * 64 + b * 16 + c * 4 + d) % n;
                        total += in[idx];
                    }
                }
            }
        }
        out[0] = total;
    }
}

// Triangular loop: O(n^2/2)
__global__ void triangular_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                sum += in[i] * in[j];
            }
        }
        out[0] = sum;
    }
}
