// Probe: complex pointer arithmetic patterns —
// pointer comparison, stepped pointer iteration,
// pointer returned from device function

__device__ float* offset_ptr(float *base, int i) {
    return base + i;
}

__global__ void device_ptr_return(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = offset_ptr(in, tid);
        out[tid] = *p * 2.0f;
    }
}

// Pointer comparison (p < q)
__global__ void ptr_compare(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = &a[tid];
        float *q = &b[tid];
        out[tid] = (p < q) ? a[tid] : b[tid];
    }
}

// Stepped iteration via pointer arithmetic
__global__ void stride2_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        float *p = in + tid * 2;
        out[tid] = p[0] + p[1];
    }
}
