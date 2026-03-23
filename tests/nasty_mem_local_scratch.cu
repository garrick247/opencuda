// Use local scratch via printf valist (.local memory), fill with values, read back.
// Tests local address computation — printf triggers .local allocation for the valist.
__global__ void nasty_mem_local_scratch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid * 4 + 0];
        int b = in[tid * 4 + 1];
        int c = in[tid * 4 + 2];
        int d = in[tid * 4 + 3];
        int sum = a + b + c + d;
        out[tid] = sum;
        if (tid == 0) {
            printf("%d %d %d %d\n", a, b, c, d);
        }
    }
}
