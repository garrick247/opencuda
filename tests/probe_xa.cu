// Probe: void* casting patterns, byte-level memory operations,
// multi-precision integer arithmetic, and complex shared memory indexing.

// ------------------------------------------------------------------
// memset-like: fill array with a value (byte level via int).

__global__ void fill_int(int *arr, int val, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = val;
}

// ------------------------------------------------------------------
// void* typed parameter (common in generic APIs).

__global__ void generic_copy(void *dst, void *src, int bytes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < bytes) {
        ((unsigned char *)dst)[tid] = ((unsigned char *)src)[tid];
    }
}

// ------------------------------------------------------------------
// Byte extraction from 32-bit word.

__device__ unsigned char extract_byte(unsigned int word, int byte_idx) {
    return (unsigned char)((word >> (byte_idx * 8)) & 0xFF);
}

__global__ void byte_extract(unsigned char *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int w = in[tid / 4];
        out[tid] = extract_byte(w, tid & 3);
    }
}

// ------------------------------------------------------------------
// 128-bit add simulated via two 64-bit halves.

__device__ void add128(unsigned long long *hi, unsigned long long *lo,
                        unsigned long long ah, unsigned long long al,
                        unsigned long long bh, unsigned long long bl) {
    unsigned long long lo_sum = al + bl;
    unsigned long long carry  = (lo_sum < al) ? 1ULL : 0ULL;
    *lo = lo_sum;
    *hi = ah + bh + carry;
}

__global__ void add128_kernel(unsigned long long *out_hi, unsigned long long *out_lo,
                               unsigned long long *ah, unsigned long long *al,
                               unsigned long long *bh, unsigned long long *bl, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        add128(&out_hi[tid], &out_lo[tid],
               ah[tid], al[tid], bh[tid], bl[tid]);
    }
}

// ------------------------------------------------------------------
// Rotate left / rotate right (32-bit).

__device__ unsigned int rotl32(unsigned int v, int n) {
    return (v << n) | (v >> (32 - n));
}

__device__ unsigned int rotr32(unsigned int v, int n) {
    return (v >> n) | (v << (32 - n));
}

__global__ void rotate_kernel(unsigned int *out, unsigned int *in, int *shifts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = shifts[tid] & 31;
        out[tid] = rotl32(in[tid], s) ^ rotr32(in[tid], s);
    }
}

// ------------------------------------------------------------------
// FNV-1a hash.

__device__ unsigned int fnv1a(unsigned char *data, int len) {
    unsigned int hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= (unsigned int)data[i];
        hash *= 16777619u;
    }
    return hash;
}

__global__ void hash_kernel(unsigned int *out, unsigned char *data, int elem_size, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fnv1a(data + tid * elem_size, elem_size);
    }
}

// ------------------------------------------------------------------
// Shared memory as lookup table.

__global__ void shared_lut(int *out, int *in, int n) {
    __shared__ int lut[16];
    int tid = threadIdx.x;

    // Thread 0..15 populate the LUT
    if (tid < 16) {
        lut[tid] = tid * tid;  // squares LUT
    }
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        out[gid] = lut[in[gid] & 15];
    }
}

// ------------------------------------------------------------------
// Loop over global array with shared-memory tiling.

__global__ void tiled_dot(float *out, float *a, float *b, int n) {
    __shared__ float sa[32], sb[32];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile * 32 < n; tile++) {
        int gid = tile * 32 + tid;
        sa[tid] = (gid < n) ? a[gid] : 0.0f;
        sb[tid] = (gid < n) ? b[gid] : 0.0f;
        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += sa[k] * sb[k];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sum;
    }
}

// ------------------------------------------------------------------
// Masked store: only store if predicate register is true.

__global__ void masked_store(float *out, float *in, float threshold, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        if (v > threshold) {
            out[tid] = v;
        }
    }
}
