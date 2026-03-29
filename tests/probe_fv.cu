// Probe: multiple kernels with shared device function that has side effects
// (writes to global memory via pointer arg), called from multiple kernels

__device__ void scatter_write(float *dst, int idx, float val, int n) {
    if (idx >= 0 && idx < n) {
        dst[idx] = val;
    }
}

__device__ float gather_read(const float *src, int idx, int n, float default_val) {
    if (idx >= 0 && idx < n) {
        return src[idx];
    }
    return default_val;
}

__global__ void scatter_kernel(float *out, int *indices, float *vals, int n, int m) {
    int tid = threadIdx.x;
    if (tid < n) {
        scatter_write(out, indices[tid], vals[tid], m);
    }
}

__global__ void gather_kernel(float *out, const float *src, int *indices, int n, int m) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = gather_read(src, indices[tid], m, -1.0f);
    }
}

__global__ void scatter_gather(float *out, float *in, int *fwd_idx, int *inv_idx,
                                 int n, int m) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = gather_read(in, inv_idx[tid], m, 0.0f);
        scatter_write(out, fwd_idx[tid], v * 2.0f, m);
    }
}
