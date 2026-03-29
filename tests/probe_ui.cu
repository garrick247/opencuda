// Probe: memory fence intrinsics, 64-bit and float atomics, clock,
// and union type punning.

// ------------------------------------------------------------------
// __threadfence variants.

__global__ void fence_ops(int *flag, int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = tid * 2;
        __threadfence();         // device-wide fence
        if (tid == 0) {
            atomicAdd(flag, 1);
        }
    }
}

__global__ void fence_block_ops(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = in[tid] + 1;
        __threadfence_block();   // block-scoped fence
        out[tid] = smem[(tid + 1) % n];
    }
}

// ------------------------------------------------------------------
// 64-bit atomic operations.

__global__ void atomic64(long long *out, long long *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        atomicAdd(&out[0], v);
        atomicMax(&out[1], v);
        atomicMin(&out[2], v);
    }
}

// ------------------------------------------------------------------
// Float atomic add.

__global__ void atomic_float_add(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(&out[tid % 32], in[tid]);
    }
}

// ------------------------------------------------------------------
// Unsigned 32-bit atomics: CAS and exchange.

__global__ void atomic_cas_exchange(unsigned int *out, unsigned int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int old = atomicExch(&out[0], in[tid]);
        // Try to CAS back to old value
        atomicCAS(&out[0], in[tid], old);
        out[tid + 1] = old;
    }
}

// ------------------------------------------------------------------
// __clock and __clock64 for timing.

__global__ void clock_kernel(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long start = clock64();
        int acc = 0;
        for (int i = 0; i < 100; i++) acc += i;
        long long end = clock64();
        out[tid] = end - start + acc;
    }
}

// ------------------------------------------------------------------
// Union for type punning (float bits ↔ int).

union FloatBits {
    float f;
    unsigned int u;
};

__global__ void float_bits_pun(unsigned int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        FloatBits fb;
        fb.f = in[tid];
        // Extract sign, exponent, mantissa
        unsigned int bits = fb.u;
        unsigned int sign = bits >> 31;
        unsigned int exp  = (bits >> 23) & 0xFF;
        unsigned int mant = bits & 0x7FFFFF;
        out[tid * 3]     = sign;
        out[tid * 3 + 1] = exp;
        out[tid * 3 + 2] = mant;
    }
}
