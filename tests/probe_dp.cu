// Probe: Boundary conditions in parse_module
// - Two struct defs with cross-references
// - Struct that uses a typedef'd struct as a field type
// - Multiple device functions with same return type
// - Global variable array assignment at module level (initializer)

struct Color {
    unsigned char r, g, b, a;
};

struct Pixel {
    int x, y;
    Color color;
};

__device__ float color_luminance(Color c) {
    return 0.299f * (float)c.r + 0.587f * (float)c.g + 0.114f * (float)c.b;
}

__device__ int pixel_is_bright(Pixel p, float threshold) {
    return color_luminance(p.color) > threshold ? 1 : 0;
}

__global__ void brightness_filter(int *out, Pixel *pixels, float threshold, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = pixel_is_bright(pixels[tid], threshold);
    }
}

__global__ void extract_luma(float *out, Pixel *pixels, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = color_luminance(pixels[tid].color);
    }
}
