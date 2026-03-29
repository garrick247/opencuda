// Probe: enum types, typedef, inline function
// - typedef int myint_t
// - enum with values
// - typedef struct
// - enum used in switch

typedef int myint_t;
typedef unsigned int uint32;

typedef struct {
    float r, g, b, a;
} Color;

enum ChannelMode {
    MODE_RED = 0,
    MODE_GREEN = 1,
    MODE_BLUE = 2,
    MODE_ALPHA = 3
};

__device__ float extract_channel(Color c, int mode) {
    if (mode == MODE_RED)   return c.r;
    if (mode == MODE_GREEN) return c.g;
    if (mode == MODE_BLUE)  return c.b;
    return c.a;
}

__global__ void color_extract(float *out, Color *colors, int mode, int n) {
    myint_t tid = (myint_t)threadIdx.x;
    if (tid < n) {
        out[tid] = extract_channel(colors[tid], mode);
    }
}

__global__ void enum_switch(int *out, int *modes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        switch (modes[tid]) {
            case MODE_RED:   out[tid] = 0; break;
            case MODE_GREEN: out[tid] = 1; break;
            case MODE_BLUE:  out[tid] = 2; break;
            default:         out[tid] = 3; break;
        }
    }
}
