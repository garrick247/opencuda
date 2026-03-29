// Probe: C++ namespace syntax (namespace should be skipped/ignored),
// using declarations, :: scope resolution

// Namespace should be silently skipped
namespace cuda_utils {

__device__ float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

}  // namespace cuda_utils

__global__ void namespace_test(float *out, float *a, float *b, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = cuda_utils::lerp(a[tid], b[tid], t[tid]);
    }
}

__global__ void clamp_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = cuda_utils::clamp_i(in[tid], 0, 255);
    }
}
