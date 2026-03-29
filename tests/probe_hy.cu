// Probe: const global arrays, __constant__ memory,
// __restrict__ parameter qualifier, pointer-to-pointer double-deref

__constant__ float c_weights[16];
__constant__ int   c_offsets[8];

__global__ void const_lookup(float *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 15;
        out[tid] = c_weights[i];
    }
}

__global__ void restrict_add(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b,
                              int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// Pointer-to-pointer: load a row pointer, then index into it
__global__ void ptr2ptr(float *out, float **rows, int col, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *row = rows[tid];
        out[tid] = row[col];
    }
}

// const local array (stack array, compile-time initialized)
__global__ void const_lut(float *out, int *idx, int n) {
    const float lut[4] = {1.0f, 2.0f, 4.0f, 8.0f};
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 3;
        out[tid] = lut[i];
    }
}
