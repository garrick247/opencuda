__global__ void printf_test(int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("hello n=%d\n", n);
    }
}
