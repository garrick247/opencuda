// Probe: Edge cases in the preprocessor
// - Macro argument with comma (problematic for simple macro parsers)
// - Macro that expands to another macro call
// - Recursive-looking macro (non-recursive by C rules)
// - Macro used in type position
// - #define INFINITY (1.0f/0.0f) - constant expression in define

#define PI 3.14159265f
#define TWO_PI (2.0f * PI)
#define HALF_PI (PI * 0.5f)

#define DEG_TO_RAD(deg) ((deg) * PI / 180.0f)
#define RAD_TO_DEG(rad) ((rad) * 180.0f / PI)

__global__ void angle_convert(float *out_deg, float *out_rad,
                               float *in_deg, float *in_rad, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_rad[tid] = DEG_TO_RAD(in_deg[tid]);
        out_deg[tid] = RAD_TO_DEG(in_rad[tid]);
    }
}

#define SQ(x) ((x)*(x))
#define HYPOT(a, b) sqrtf(SQ(a) + SQ(b))

__global__ void hypot_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = HYPOT(a[tid], b[tid]);
    }
}
