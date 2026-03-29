// Probe: type-punning casts, reinterpret via pointer cast,
// int-to-float bit cast (via union-style or pointer),
// chained casts (int)(float)(double)x

__device__ unsigned int float_to_bits(float f) {
    return *((unsigned int *)&f);
}

__device__ float bits_to_float(unsigned int u) {
    return *((float *)&u);
}

__global__ void pun_roundtrip(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int bits = float_to_bits(in[tid]);
        // flip sign bit
        bits ^= 0x80000000u;
        out[tid] = bits_to_float(bits);
    }
}

// Chained casts
__global__ void chain_cast(int *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (int)(float)(in[tid]);
    }
}

// Cast pointer through void*
__global__ void void_ptr_cast(void *out_raw, float *in, int n) {
    float *out = (float *)out_raw;
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}
