// Probe: __ldg (load via texture cache), __ldcg, __ldcs, __ldca,
// explicit cache hints in loads — all map to ld.global.nc or ld variants

__global__ void ldg_kernel(float *out, const float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // __ldg should emit ld.global.nc
        float v = __ldg(&in[tid]);
        out[tid] = v * 2.0f;
    }
}

__global__ void ldg_int(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = __ldg(&in[tid]);
        out[tid] = v + 1;
    }
}

// __ldcg = cache at global level (streaming)
__global__ void ldcg_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = __ldcg(&in[tid]);
        out[tid] = v;
    }
}
