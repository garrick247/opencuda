// Probe: explicit type punning via union, union with multiple members
// accessed after write to different member (should parse gracefully)

union FloatInt {
    float f;
    unsigned int u;
    int i;
};

__device__ unsigned int float_to_bits(float f) {
    FloatInt fi;
    fi.f = f;
    return fi.u;
}

__device__ float bits_to_float(unsigned int u) {
    FloatInt fi;
    fi.u = u;
    return fi.f;
}

__global__ void bit_trick(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Fast abs via bit manipulation
        unsigned int bits = float_to_bits(v);
        bits &= 0x7FFFFFFF;  // clear sign bit
        out[tid] = bits_to_float(bits);
    }
}

// Simple packing via bit manipulation
__global__ void pack_unpack(unsigned int *out, unsigned int *lo_in,
                              unsigned int *hi_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int packed = (hi_in[tid] << 16) | (lo_in[tid] & 0xFFFF);
        out[tid] = packed;
    }
}
