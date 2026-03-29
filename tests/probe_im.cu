// Probe: compound assignments in for-increment (+=, -=)
// when variables are bound to expression-result Values

__global__ void step_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = n;   // a = Value(param_n, not "a")  -> actually a = n reuses param val
        int b = n * 2;  // b = computed Value
        int sum = 0;
        // j is initialized by expression, and uses compound assign in increment
        for (int i = 0, j = n - 1; i < j; i += 2, j -= 3) {
            sum += i + j;
        }
        out[0] = sum;
    }
}

// Multiple compound updates in loop
__global__ void multi_compound(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = n + 5;
        int y = n - 3;
        int z = n * 2;
        for (int iter = 0; iter < 10; iter++) {
            x += y;
            y -= z;
            z ^= x;
        }
        out[0] = x;
        out[1] = y;
        out[2] = z;
    }
}

// Compound assign on ptr-valued variable (should do load-op-store)
__global__ void ptr_compound(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // These compound assigns work on scalar values
        v *= 2.0f;
        v += 1.0f;
        v -= 0.5f;
        out[tid] = v;
    }
}
