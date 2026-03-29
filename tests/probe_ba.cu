// Probe: C++ features sometimes used in CUDA device code
// - template-like macros
// - inline keyword (should be ignored)
// - __forceinline__ (should be treated like __device__)
// - extern "C" block (should be skipped/parsed)
// - namespace usage (should fail gracefully or be skipped)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(v, lo, hi) MAX(MIN(v, hi), lo)

__forceinline__ __device__ float safe_sqrt(float x) {
    return sqrtf(x > 0.0f ? x : 0.0f);
}

__global__ void macro_clamp(float *out, float *in, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = CLAMP(in[tid], lo, hi);
    }
}

__global__ void forceinline_use(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_sqrt(in[tid]);
    }
}
