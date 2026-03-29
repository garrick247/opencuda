// Probe: unusual for-loop init forms
// - for(;;) with variable declared before loop
// - for with pre-declared counter: for (i = 0; ...)  (no type, just assign)
// - for loop where body modifies the condition variable from a different path
// - for loop with COMMA in condition: for(a = 0, b = 0; a < n && b < m; a++, b += 2)

__global__ void pre_decl_for(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int i;
        for (i = 0; i < 8; i++) {
            sum += in[(tid + i) % n];
        }
        out[tid] = sum;
    }
}

__global__ void for_comma_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int a, b;
        for (a = 0, b = 0; a < n && b < n; a++, b += 2) {
            sum += in[a] + in[b % n];
        }
        out[tid] = sum + tid;
    }
}

// While loop that modifies index
__global__ void stride_while(float *out, float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        int i = tid;
        while (i < n) {
            sum += in[i];
            i += stride;
        }
        out[tid] = sum;
    }
}
