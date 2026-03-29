// Probe: array of structs access pattern, struct returned via pointer param,
// nested struct access, struct field arithmetic

typedef struct {
    float x, y;
} Vec2;

typedef struct {
    int r, g, b, a;
} Color;

typedef struct {
    Vec2 pos;    // NOT nested struct syntax — two separate floats
    float scale;
} Transform;

// Array of structs: load/store patterns
__global__ void aos_scale(Vec2 *out, Vec2 *in, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid].x * scale;
        float y = in[tid].y * scale;
        out[tid].x = x;
        out[tid].y = y;
    }
}

// Color struct: 4-component RGBA
__global__ void color_blend(Color *out, Color *a, Color *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid].r = (a[tid].r + b[tid].r) / 2;
        out[tid].g = (a[tid].g + b[tid].g) / 2;
        out[tid].b = (a[tid].b + b[tid].b) / 2;
        out[tid].a = (a[tid].a + b[tid].a) / 2;
    }
}

// Struct field in loop accumulation
__global__ void struct_sum(Vec2 *out, Vec2 *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sx = 0.0f;
        float sy = 0.0f;
        for (int i = 0; i < n; i++) {
            sx += in[i].x;
            sy += in[i].y;
        }
        out[0].x = sx;
        out[0].y = sy;
    }
}

// Struct used as index computation base
__global__ void struct_index(int *out, Color *colors, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Compute brightness from RGB
        int bright = colors[tid].r + colors[tid].g + colors[tid].b;
        out[tid] = bright;
    }
}
