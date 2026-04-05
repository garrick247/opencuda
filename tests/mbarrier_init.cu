// Test: mbarrier intrinsics
__global__ void kernel_mbarrier(long long mbar, int count) {
    __mbarrier_init(mbar, count);
    int result = __mbarrier_try_wait_parity(mbar, 0);
}
