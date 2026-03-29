// Probe: atomics on various types — atomicMin/Max on int/uint,
// atomicCAS pattern, atomicExch, atomicOr/And/Xor

__global__ void atomic_minmax(int *out_min, int *out_max, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicMin(&out_min[0], in[tid]);
        atomicMax(&out_max[0], in[tid]);
    }
}

__global__ void atomic_cas_lock(int *lock, float *data, float val, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Spin-wait lock acquire pattern (CAS loop)
        int old = atomicCAS(lock, 0, 1);
        if (old == 0) {
            data[0] += val;
            atomicExch(lock, 0);
        }
    }
}

__global__ void atomic_bitwise(unsigned int *flags, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int bit = 1u << (in[tid] & 31);
        atomicOr(&flags[0], bit);
        atomicAnd(&flags[1], ~bit);
        atomicXor(&flags[2], bit);
    }
}

__global__ void atomic_float(float *sum, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(sum, in[tid]);
    }
}
