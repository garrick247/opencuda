// Probe: atomics with all flavors — add, min, max, and, or, xor, exch, cas,
// both int and float atomics, __shared__ atomics

__global__ void atomic_all(int *out_add, int *out_max, int *out_min,
                            unsigned int *out_and, unsigned int *out_or,
                            int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        atomicAdd(out_add, val);
        atomicMax(out_max, val);
        atomicMin(out_min, val);
        atomicAnd(out_and, (unsigned int)val);
        atomicOr(out_or, (unsigned int)val);
    }
}

// Float atomicAdd
__global__ void float_atomic(float *sum, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(sum, in[tid]);
    }
}

// atomicExch and atomicCAS
__global__ void exch_cas(int *lock, int *data, int new_val, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Spinlock acquire
        int old;
        do {
            old = atomicCAS(lock, 0, 1);
        } while (old != 0);
        // Critical section
        *data = new_val;
        // Release
        atomicExch(lock, 0);
    }
}

// Shared memory atomics
__global__ void shared_histogram(int *out, int *in, int n) {
    __shared__ int hist[16];
    int tid = threadIdx.x;
    if (tid < 16) hist[tid] = 0;
    __syncthreads();
    if (tid < n) {
        int bucket = in[tid] & 15;
        atomicAdd(&hist[bucket], 1);
    }
    __syncthreads();
    if (tid < 16) out[tid] = hist[tid];
}
