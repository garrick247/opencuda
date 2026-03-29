// Probe: Patterns that appear in real ML/vision CUDA kernels
// - 2D convolution indexing
// - Boundary check with multiple conditions
// - Accumulate over multiple memory ranges
// - Double-buffered shared memory pattern
// - Loop with complex bounds that include thread/block indices

#define TILE_W 16
#define TILE_H 16

__global__ void conv2d_naive(float *out, float *in, float *kernel,
                              int H, int W, int kH, int kW) {
    int ox = threadIdx.x + blockIdx.x * TILE_W;
    int oy = threadIdx.y + blockIdx.y * TILE_H;
    if (ox >= W || oy >= H) return;

    float sum = 0.0f;
    int kH2 = kH / 2;
    int kW2 = kW / 2;
    for (int ky = -kH2; ky <= kH2; ky++) {
        for (int kx = -kW2; kx <= kW2; kx++) {
            int iy = oy + ky;
            int ix = ox + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                float v = in[iy * W + ix];
                float k = kernel[(ky + kH2) * kW + (kx + kW2)];
                sum += v * k;
            }
        }
    }
    out[oy * W + ox] = sum;
}
