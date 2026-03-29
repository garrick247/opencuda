// Probe: Unusual global-level patterns
// - __device__ function defined after __global__ that uses it
// - __device__ variable defined inside a __global__ kernel (nope — module level only)
// - Struct definition inside a function (C++ only — should fail or skip)
// - Function with asm() statement (should skip/fail gracefully)
// - Multiple initializer levels in __constant__

__constant__ float c_kernel[9] = {
    0.0625f, 0.125f, 0.0625f,
    0.125f,  0.25f,  0.125f,
    0.0625f, 0.125f, 0.0625f
};

__global__ void gaussian_blur(float *out, float *in, int W, int H) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x <= 0 || x >= W - 1 || y <= 0 || y >= H - 1) return;
    
    float sum = 0.0f;
    int k = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            sum += in[(y + dy) * W + (x + dx)] * c_kernel[k];
            k++;
        }
    }
    out[y * W + x] = sum;
}
