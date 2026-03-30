// Probe: obscure but valid C that might break — nested struct literal assignment,
// ternary returning different pointer types, function call as array index,
// complex comma expressions, chained arrow operator, cast in ternary condition,
// double negation (!!), bitfield-like manual packing, and volatile shared
// memory in reduction (prevents optimization).

// ------------------------------------------------------------------
// Double negation (!!) to convert to bool.

__global__ void double_neg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = !!in[tid];  // 0 → 0, nonzero → 1
}

// ------------------------------------------------------------------
// Function call result as array index.

__device__ int clamp_idx(int idx, int max) {
    return (idx < 0) ? 0 : (idx >= max) ? max - 1 : idx;
}

__global__ void call_as_index(int *out, int *in, int *indices, int max_idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[clamp_idx(indices[tid], max_idx)];
    }
}

// ------------------------------------------------------------------
// Manual bit-packing: pack 4 bytes into one int.

__global__ void pack4(unsigned *out, unsigned char *b0, unsigned char *b1,
                         unsigned char *b2, unsigned char *b3, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned packed = ((unsigned)b3[tid] << 24) |
                          ((unsigned)b2[tid] << 16) |
                          ((unsigned)b1[tid] <<  8) |
                          ((unsigned)b0[tid]);
        out[tid] = packed;
    }
}

// ------------------------------------------------------------------
// Unpack int into 4 bytes.

__global__ void unpack4(unsigned char *b0, unsigned char *b1,
                           unsigned char *b2, unsigned char *b3,
                           unsigned *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned v = in[tid];
        b0[tid] = (unsigned char)(v & 0xFF);
        b1[tid] = (unsigned char)((v >> 8) & 0xFF);
        b2[tid] = (unsigned char)((v >> 16) & 0xFF);
        b3[tid] = (unsigned char)((v >> 24) & 0xFF);
    }
}

// ------------------------------------------------------------------
// Volatile shared memory reduction (forces re-reads).

__global__ void volatile_reduce(float *out, float *in, int n) {
    volatile __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    // Last warp does reduction without __syncthreads (volatile ensures visibility)
    if (tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) {
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Complex expression: nested function calls + arithmetic + ternary.

__device__ float safe_sqrt(float x) { return (x > 0.0f) ? sqrtf(x) : 0.0f; }
__device__ float safe_log(float x)  { return (x > 0.0f) ? logf(x)  : -1e30f; }
__device__ float safe_div(float a, float b) { return (b != 0.0f) ? a / b : 0.0f; }

__global__ void safe_math_chain(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float va = a[gid], vb = b[gid];
        // safe_sqrt(safe_div(a, b)) + safe_log(a * b)
        out[gid] = safe_sqrt(safe_div(va, vb)) + safe_log(va * vb);
    }
}

// ------------------------------------------------------------------
// Multiple assignment targets from one expression chain.

__global__ void multi_assign(int *out1, int *out2, int *out3,
                                int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * 2;
        int b = a + 1;
        int c = b * b;
        out1[tid] = a;
        out2[tid] = b;
        out3[tid] = c;
    }
}

// ------------------------------------------------------------------
// Warp-divergent early return with later convergence.

__global__ void diverge_converge(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int lane = threadIdx.x & 31;
    float v = in[gid];

    // First 16 lanes do one thing, last 16 do another
    float result;
    if (lane < 16) {
        result = v * v;
    } else {
        result = sqrtf(fabsf(v));
    }

    // All lanes converge here: warp reduce
    result += __shfl_xor_sync(0xFFFFFFFF, result, 16);
    result += __shfl_xor_sync(0xFFFFFFFF, result, 8);
    result += __shfl_xor_sync(0xFFFFFFFF, result, 4);
    result += __shfl_xor_sync(0xFFFFFFFF, result, 2);
    result += __shfl_xor_sync(0xFFFFFFFF, result, 1);

    out[gid] = result;
}
