// Test: TMA cp.async.bulk intrinsics
__global__ void kernel_tma_1d(long long smem_ptr, long long desc, int coord, long long mbar) {
    __cp_async_bulk_tensor_1d(smem_ptr, desc, coord, mbar);
    __cp_async_bulk_commit_group();
    __cp_async_bulk_wait_group(0);
}

__global__ void kernel_tma_2d(long long smem_ptr, long long desc, int x, int y, long long mbar) {
    __cp_async_bulk_tensor_2d(smem_ptr, desc, x, y, mbar);
    __cp_async_bulk_commit_group();
    __cp_async_bulk_wait_group(0);
}
