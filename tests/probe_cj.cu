// Probe: Tricky memory patterns
// - Store to dereferenced pointer from function return
// - Pointer to local struct field passed to device func
// - Global struct array with field access
// - Nested dereference: **ptr

struct Config {
    int width;
    int height;
    float scale;
    int flags;
};

__device__ void apply_scale(float *p, float scale) {
    *p *= scale;
}

__device__ void init_config(Config *cfg, int w, int h, float s) {
    cfg->width = w;
    cfg->height = h;
    cfg->scale = s;
    cfg->flags = 0;
}

__global__ void config_kernel(float *out, Config *configs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Config cfg;
        init_config(&cfg, 640, 480, 1.5f);
        float v = (float)(tid % cfg.width) / (float)cfg.width;
        apply_scale(&v, cfg.scale);
        out[tid] = v + (float)configs[tid].flags;
    }
}
