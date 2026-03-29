// Regression: local (stack) array declarations in CUDA kernels.
// float buf[N] allocates PTX .local memory; buf[i] computes address
// and uses ld.local/st.local instructions.
// Without fix: ParseError "expected SEMI, got '['" on array declarations.
__global__ void local_array2_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float buf[4];
        // Load into local array
        buf[0] = in[tid];
        buf[1] = buf[0] * 2.0f;
        buf[2] = buf[0] + buf[1];
        buf[3] = buf[2] / 3.0f;
        // Read back from local array
        out[tid] = buf[3];
    }
}

__global__ void local_int_array_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int scratch[8];
        // Scatter-gather using local array
        for (int i = 0; i < 8; i++) {
            scratch[i] = in[tid] + i;
        }
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += scratch[i];
        }
        out[tid] = sum;
    }
}
