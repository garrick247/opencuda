// Regression: union types and __constant__ memory.
// Without fix:
//   union FloatInt { ... };  → ParseError "undefined variable 'union'"
//   __constant__ float arr[4]; → ParseError "undefined variable 'scale_factors'"

union FloatInt { float f; unsigned int i; };

union Vec2u { float x; float y; };

__constant__ float scale_factors[4];
__constant__ int lut[256];

__global__ void union_test(unsigned int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union FloatInt u;
        u.f = in[tid];
        // Note: in our model union fields are separate registers, so
        // u.i won't contain the float bits — but PTX must compile.
        out[tid] = u.i;
    }
}

__global__ void const_mem_test(float *out, float *in, int n, int channel) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * scale_factors[channel];
    }
}

__global__ void const_lut_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = in[tid] & 0xFF;
        out[tid] = lut[idx];
    }
}
