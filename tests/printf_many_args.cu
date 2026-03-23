__global__ void printf_many_args(int *data) {
    if (threadIdx.x == 0) {
        int a = data[0];
        int b = data[1];
        int c = data[2];
        int d = data[3];
        printf("a=%d b=%d c=%d d=%d\n", a, b, c, d);
    }
}
