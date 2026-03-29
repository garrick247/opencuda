// Probe: __syncthreads_count, __syncthreads_and, __syncthreads_or
// (barrier + reduction variants), __threadfence, __threadfence_block

__global__ void sync_count(int *out, int *pred, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int p = (pred[tid] > 0) ? 1 : 0;
        int cnt = __syncthreads_count(p);
        out[tid] = cnt;
    }
}

__global__ void sync_and(int *out, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int f = flags[tid];
        int all = __syncthreads_and(f);
        out[tid] = all;
    }
}

__global__ void sync_or(int *out, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int f = flags[tid];
        int any = __syncthreads_or(f);
        out[tid] = any;
    }
}

// __threadfence as a statement
__global__ void fence_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid];
        __threadfence();
    }
}
