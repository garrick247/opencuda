// Probe: __restrict__ pointer with pointer arithmetic, void* casts,
// sizeof in array bounds, sizeof on expressions

__global__ void ptr_arith_restrict(float * __restrict__ out,
                                    const float * __restrict__ in,
                                    int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        const float *p = in + tid * stride;
        float sum = 0.0f;
        for (int i = 0; i < stride; i++) {
            sum += p[i];
        }
        out[tid] = sum;
    }
}

// sizeof expression in index computation
__global__ void sizeof_index(char *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sizeof(int) == 4
        int byte_off = tid * (int)sizeof(int);
        out[byte_off]     = (char)(in[tid] & 0xFF);
        out[byte_off + 1] = (char)((in[tid] >> 8) & 0xFF);
        out[byte_off + 2] = (char)((in[tid] >> 16) & 0xFF);
        out[byte_off + 3] = (char)((in[tid] >> 24) & 0xFF);
    }
}

// Pointer difference (ptrdiff)
__global__ void ptr_diff(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Distance in elements (not bytes) — treat pointer subtraction as integer
        int dist = (int)(b - a);
        out[tid] = a[tid] + dist;
    }
}
