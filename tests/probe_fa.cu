// Probe: complex type expressions — array of pointers, pointer to array,
// multi-level pointer, typedef chains

typedef float* FloatPtr;
typedef FloatPtr* FloatPtrPtr;
typedef int (*IntArrayPtr)[4];

__global__ void float_ptr_kernel(FloatPtr out, FloatPtr in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

// Pointer-to-pointer (float**)
__global__ void ptr_ptr_deref(float *out, float **rows, int row, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = rows[row][tid];
    }
}

// Multi-level typedef
typedef unsigned int uint32;
typedef uint32 u32;

__global__ void typedef_chain(u32 *out, u32 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        u32 v = in[tid];
        v = (v ^ (v >> 16)) * 0x45d9f3bU;
        v = (v ^ (v >> 16)) * 0x45d9f3bU;
        v = v ^ (v >> 16);
        out[tid] = v;
    }
}
