// Probe: Complex patterns in parse_module (module-level declarations)
// - Multiple __constant__ arrays of different types
// - __device__ global variable arrays
// - Multiple typedefs including struct typedefs
// - Forward declaration (struct before definition)

typedef float float4_t[4];
typedef int ivec2[2];

__constant__ float c_table_f[16];
__constant__ int c_table_i[16];
__device__ float g_scale_factor = 1.0f;
__device__ int g_max_iter = 100;

__global__ void multi_const(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int key = in[tid] & 15;
        float fv = c_table_f[key];
        int iv = c_table_i[key];
        out[tid] = fv * (float)iv * g_scale_factor;
    }
}

__global__ void iter_limit(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int iter = 0;
        while (iter < g_max_iter && v > 0.001f) {
            v = v * 0.9f;
            iter++;
        }
        out[tid] = v + (float)iter;
    }
}
