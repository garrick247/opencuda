// Regression: printf with negative int32 argument must use cvt.s64.s32 (sign extension),
// not cvt.u64.u32 (zero extension) when packing the valist slot.
// Without sign extension, printf("%d", -1) would print 4294967295 instead of -1.
__global__ void printf_negative(int *data) {
    if (threadIdx.x == 0) {
        int neg = data[0];   // expect negative value (e.g. -1)
        int pos = data[1];   // expect positive
        printf("neg=%d pos=%d\n", neg, pos);
    }
}
