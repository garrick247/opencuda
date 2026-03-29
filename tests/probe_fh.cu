// Probe: complex pointer expression — pointer + int64 offset,
// pointer comparison, null pointer check pattern

__global__ void null_check_pattern(float *out, float *in, int *valid, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (valid[tid]) {
            out[tid] = in[tid];
        } else {
            out[tid] = 0.0f;
        }
    }
}

// Pointer passed through multiple local variables
__global__ void ptr_alias(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *src = in + tid;
        float *dst = out + tid;
        float v = *src;
        *dst = v * v;
    }
}

// Pointer arithmetic with 64-bit index
__global__ void ptr_large_offset(float *out, float *in, long long stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long idx = (long long)tid * stride;
        out[tid] = in[idx];
    }
}

// Array of pointers
__global__ void ptr_array(float *out, float **ptrs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = ptrs[tid];
        out[tid] = p[0] + p[1];
    }
}
