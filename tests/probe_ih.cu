// Probe: string literal in printf, printf with various format specifiers,
// printf return value (ignored), multiple printfs in same kernel

__global__ void debug_print(float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("n = %d\n", n);
        for (int i = 0; i < n && i < 4; i++) {
            printf("in[%d] = %f\n", i, in[i]);
        }
    }
}

__global__ void conditional_print(int *in, int n, int threshold) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (in[tid] > threshold) {
            printf("thread %d: value %d exceeds threshold %d\n",
                   tid, in[tid], threshold);
        }
    }
}

__global__ void print_types(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int u = 42u;
        long long ll = 1000000LL;
        float f = 3.14f;
        printf("u=%u ll=%lld f=%f\n", u, ll, f);
        *out = 1;
    }
}
