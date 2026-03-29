// Probe: union types, __noinline__ device functions, memory fence
// variants, __constant__ array in complex access pattern.

// ------------------------------------------------------------------
// Union: float/int reinterpretation.

union FloatInt {
    float f;
    int   i;
};

__global__ void float_bits(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union FloatInt u;
        u.f = in[tid];
        // Read bits as int
        out[tid] = u.i;
    }
}

__global__ void bits_to_float(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union FloatInt u;
        u.i = in[tid];
        out[tid] = u.f;
    }
}

// ------------------------------------------------------------------
// __noinline__ device function — parsed, behavior same as regular device fn.

__device__ __noinline__ int noinline_fn(int v, int k) {
    return v * k + k * k;
}

__global__ void uses_noinline(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = noinline_fn(in[tid], 3);  // v*3 + 9
    }
}

// ------------------------------------------------------------------
// __constant__ array used with non-trivial index expression.

__constant__ float coeffs[16];

__global__ void const_indexed(float *out, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid] & 15;  // ensure in bounds
        float c = coeffs[idx];
        out[tid] = c * (float)(tid + 1);
    }
}

// ------------------------------------------------------------------
// __constant__ used in loop with computed index.

__constant__ int lookup[8];

__global__ void const_loop_access(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            // Index into constant array depends on both loop var and input
            acc += lookup[(v + i) & 7];
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Memory fence variants.

__global__ void fence_variants(int *out, volatile int *shared, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Write with fence
        shared[tid] = tid;
        __threadfence();         // global fence
        __threadfence_block();   // block-level fence
        // Read after fence
        int v = shared[tid];
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Union with double/long long.

union DoubleBits {
    double  d;
    long long i;
};

__global__ void double_bits(long long *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union DoubleBits u;
        u.d = in[tid];
        out[tid] = u.i;  // raw bits of the double
    }
}

// ------------------------------------------------------------------
// Complex shared memory reduction (tree-based, 256 threads).

__global__ void tree_reduce(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Binary tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}
