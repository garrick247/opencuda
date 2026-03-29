// Test: const T * __restrict__ parameters emit ld.global.nc
// Both qualifiers together → AddrSpace.CONST → non-caching read-only load
__global__ void const_restrict(const float * __restrict__ in, float * out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

// Variant: const after type name (float const * __restrict__)
__global__ void const_restrict_postfix(float const * __restrict__ in, float * out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        out[tid] = in[tid] + 1.0f;
    }
}

// Without __restrict__: const alone should NOT produce ld.global.nc (could alias)
__global__ void const_no_restrict(const float * in, float * out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        out[tid] = in[tid];
    }
}
