// Probe: Memory ordering patterns, atomics, and synchronization
// - atomicMin, atomicMax usage
// - Compound atomic: atomicAdd return value used
// - __threadfence() usage
// - Cooperative memory patterns with flags

__global__ void atomic_histogram(int *hist, int *in, int n, int nbins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int bin = in[tid] % nbins;
        atomicAdd(&hist[bin], 1);
    }
}

__global__ void atomic_minmax(int *out_min, int *out_max, int *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        atomicMin(out_min, in[tid]);
        atomicMax(out_max, in[tid]);
    }
}

// Using atomicAdd return value
__global__ void atomic_counter_use(int *out, int *counter, int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n && data[tid] > 0) {
        int slot = atomicAdd(counter, 1);
        if (slot < n) {
            out[slot] = data[tid];
        }
    }
}
