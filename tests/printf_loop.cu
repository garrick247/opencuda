// printf inside a small counted loop — exercises PrintfInst remap in unroller.
__global__ void printf_loop(int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < 4; i++) {
            printf("i=%d n=%d\n", i, n);
        }
    }
}
