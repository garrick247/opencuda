// Integer and atomic kernels for runtime validation.
// Using int signature: (int *out, int *a, int *b, int n)

// Integer add
__global__ void int_add(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] + b[gid];
}

// Integer multiply
__global__ void int_mul(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] * b[gid];
}

// Bitwise XOR
__global__ void int_xor(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] ^ b[gid];
}

// Bitwise AND
__global__ void int_and(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] & b[gid];
}

// Left shift by b[i] & 31
__global__ void int_shl(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] << (b[gid] & 31);
}

// Max of two integers
__global__ void int_max(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (a[gid] > b[gid]) ? a[gid] : b[gid];
}

// Absolute difference
__global__ void int_absdiff(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int d = a[gid] - b[gid];
        out[gid] = (d < 0) ? -d : d;
    }
}

// Popcount
__global__ void int_popc(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = __popc(a[gid]);
}

// Count leading zeros
__global__ void int_clz(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = __clz(a[gid]);
}

// Bit reverse
__global__ void int_brev(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (int)__brev((unsigned)a[gid]);
}
