// Probe: Patterns with printf and string formatting
// - printf with multiple % format specifiers
// - printf with float %f and int %d
// - printf in conditional (only one thread prints)
// - printf with expressions in arguments
// - Nested printf calls

__global__ void debug_printf(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = v * 2.0f;
        if (tid == 0) {
            printf("thread 0: in=%f out=%f\n", v, v * 2.0f);
        }
    }
}

__global__ void printf_check(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = a[tid] + b[tid];
        out[tid] = sum;
        if (tid < 4) {
            printf("tid=%d a=%d b=%d sum=%d\n", tid, a[tid], b[tid], sum);
        }
    }
}
