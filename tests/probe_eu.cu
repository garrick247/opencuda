// Probe: complex initializer lists — struct initializers, array initializers
// Also: designated initializers (C99 style: .field = value)
// Also: compound literal style

struct Config {
    int width;
    int height;
    float scale;
    int flags;
};

__constant__ Config g_config = {256, 256, 1.0f, 3};

__global__ void use_config(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int w = g_config.width;
        int h = g_config.height;
        float s = g_config.scale;
        int f = g_config.flags;
        out[tid] = in[tid] * s + (float)(w + h + f);
    }
}

// Local struct with brace initializer (C99)
__global__ void local_struct_init(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Brace-init of local struct
        Config cfg;
        cfg.width = 128;
        cfg.height = 64;
        cfg.scale = 0.5f;
        cfg.flags = 0;
        out[tid] = (float)(cfg.width + cfg.height) * cfg.scale;
    }
}

// Array initializer with fewer elements than size (rest zero-init)
__constant__ float g_lut[8] = {1.0f, 2.0f, 4.0f, 8.0f};

__global__ void lut_lookup(float *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 7;
        out[tid] = g_lut[i];
    }
}
