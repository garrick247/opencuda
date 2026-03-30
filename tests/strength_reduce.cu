// Test: strength reduction optimizations
// - unsigned div by power of 2 → shift right
// - unsigned mod by power of 2 → bitwise AND
// - mul by power of 2 → shift left (already existed)

__global__ void div_shift(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // div by 4 → shr 2; div by 8 → shr 3
        out[tid] = in[tid] / 4 + in[tid] / 8;
    }
}

__global__ void mod_mask(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // mod 16 → and 15; mod 256 → and 255
        out[tid] = (in[tid] % 16) + (in[tid] % 256);
    }
}

__global__ void mul_shift(unsigned *out, unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // mul 8 → shl 3
        out[tid] = in[tid] * 8;
    }
}
