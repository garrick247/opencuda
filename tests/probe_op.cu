// Probe: additional atomics (Max/Min/Exch/CAS), bit intrinsics (__popc/__clz/__ffs),
// __threadfence, and mixed struct+pointer device function calling.

// ------------------------------------------------------------------
// atomicMax, atomicMin, atomicExch, atomicCAS.

__global__ void atomic_variety(int *maxval, int *minval, int *exch_dst,
                                int *cas_dst, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        atomicMax(maxval, v);
        atomicMin(minval, v);
    }
    if (tid == 0) {
        atomicExch(exch_dst, 42);
        atomicCAS(cas_dst, 0, 99);   // if *cas_dst == 0, set to 99
    }
}

// ------------------------------------------------------------------
// Bit manipulation intrinsics.

__global__ void bit_intrinsics(int *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = data[tid];
        out[tid*3+0] = __popc(v);       // population count
        out[tid*3+1] = __clz(v);        // count leading zeros
        out[tid*3+2] = __ffs(v);        // find first set bit
    }
}

// ------------------------------------------------------------------
// __threadfence and __threadfence_block: memory ordering barriers.

__device__ int g_flag;

__global__ void threadfence_test(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Write data, then fence, then set flag
        out[0] = data[0];
        __threadfence();
        g_flag = 1;
    }
    __syncthreads();
    if (tid == 1 && g_flag == 1) {
        out[1] = data[1];
        __threadfence_block();
    }
}

// ------------------------------------------------------------------
// Device function with struct by value AND pointer parameter.

struct Config { float alpha; float beta; int n; };

__device__ float apply_config(Config cfg, float *data, int idx) {
    return cfg.alpha * data[idx] + cfg.beta;
}

__global__ void mixed_call(float *out, float *data, float alpha, float beta, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Config c;
        c.alpha = alpha;
        c.beta  = beta;
        c.n     = n;
        out[tid] = apply_config(c, data, tid);
    }
}
