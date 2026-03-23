// Nasty: printf inside if (tid == 0) inside a loop.
// Tests that the vprintf call block inside a conditional branch doesn't
// corrupt register state for the loop's continue path.
__global__ void debug_loop(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = data[i];
        sum = sum + v;
        if (tid == 0) {
            printf("i=%d v=%f\n", i, v);
        }
    }
    if (tid == 0) {
        printf("sum=%f\n", sum);
    }
    data[tid] = sum;
}
