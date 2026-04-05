// Test: surf1Dread, surf1Dwrite intrinsics
__global__ void kernel_surf_read(int *out, long long surfObj, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int val = surf1Dread(surfObj, idx * 4);
        out[idx] = val;
    }
}

__global__ void kernel_surf_write(long long surfObj, int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int val = data[idx];
        surf1Dwrite(val, surfObj, idx * 4);
    }
}
