// Probe: anonymous struct inside union (currently a known limitation) — skip
// Instead: complex union with named inner struct, union member access

union Color32 {
    unsigned int packed;
    float f_val;
};

struct PixelRGBA {
    unsigned char r, g, b, a;
};

__device__ Color32 pack_color(float r, float g, float b, float a) {
    Color32 c;
    c.packed = ((unsigned int)(a * 255.0f) << 24) |
               ((unsigned int)(b * 255.0f) << 16) |
               ((unsigned int)(g * 255.0f) << 8) |
               ((unsigned int)(r * 255.0f));
    return c;
}

__global__ void color_pack(unsigned int *out, float *rgba, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 4;
        Color32 c = pack_color(rgba[base], rgba[base+1], rgba[base+2], rgba[base+3]);
        out[tid] = c.packed;
    }
}

// Union accessed as float then as int
__global__ void union_reinterpret(unsigned int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Color32 c;
        c.f_val = in[tid];
        out[tid] = c.packed;  // reinterpret float bits as uint
    }
}
