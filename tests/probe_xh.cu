// Probe: advanced warp patterns, device function pointers (simulated via
// switch dispatch), complex memory access patterns, and GPU-specific
// idioms used in production CUDA code.

// ------------------------------------------------------------------
// Thread coarsening: each thread processes multiple elements.

__global__ void thread_coarsen(float *out, float *in, int n, int work_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * work_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < work_per_thread && start + i < n; i++) {
        sum += in[start + i];
    }
    if (start < n) {
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Ping-pong reduction: two __syncthreads() per step.

__global__ void ping_pong_reduce(int *out, int *in, int n) {
    __shared__ int a[128], b[128];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    a[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Step 1: a → b
    if (tid < 64) b[tid] = a[tid] + a[tid + 64];
    __syncthreads();
    // Step 2: b → a
    if (tid < 32) a[tid] = b[tid] + b[tid + 32];
    __syncthreads();
    // Step 3: warp reduce
    if (tid < 32) {
        int v = a[tid];
        v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
        if (tid == 0) out[blockIdx.x] = v;
    }
}

// ------------------------------------------------------------------
// Warp-level broadcast from lane 0.

__global__ void warp_broadcast(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        // Lane 0 of each warp broadcasts its value to all lanes
        int bcast = __shfl_sync(0xFFFFFFFF, v, 0);
        out[gid] = bcast;
    }
}

// ------------------------------------------------------------------
// Segmented scan: each segment of 32 gets its own prefix sum.

__global__ void segmented_prefix(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int lane = threadIdx.x & 31;
        // Inclusive prefix sum within warp via shfl_up
        for (int offset = 1; offset < 32; offset <<= 1) {
            int y = __shfl_up_sync(0xFFFFFFFF, v, offset);
            if (lane >= offset) v += y;
        }
        out[gid] = v;
    }
}

// ------------------------------------------------------------------
// Load-balancing: process work items from a queue.

__global__ void process_queue(int *out, int *queue, int *queue_len, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int len = *queue_len;
    if (tid < len) {
        int item = queue[tid];
        // Process item: compute some function
        out[tid] = item * item + item;
    }
}

// ------------------------------------------------------------------
// Neighbor gather: each thread reads from left and right neighbors.

__global__ void neighbor_gather(float *out, float *in, int n) {
    __shared__ float smem[256 + 2];  // +2 halo
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid + 1] = (gid < n) ? in[gid] : 0.0f;
    // Load halo elements
    if (tid == 0 && gid > 0)           smem[0]   = in[gid - 1];
    else if (tid == 0)                  smem[0]   = 0.0f;
    if (tid == blockDim.x - 1 && gid + 1 < n) smem[tid + 2] = in[gid + 1];
    else if (tid == blockDim.x - 1)    smem[tid + 2] = 0.0f;
    __syncthreads();

    if (gid > 0 && gid < n - 1) {
        out[gid] = (smem[tid] + smem[tid + 1] + smem[tid + 2]) / 3.0f;
    }
}

// ------------------------------------------------------------------
// Stream compaction scatter: write active elements to output.

__global__ void stream_compact(int *out_data, int *out_count,
                                int *in_data, int *prefix_sum, int *mask, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && mask[tid]) {
        int pos = prefix_sum[tid];
        out_data[pos] = in_data[tid];
    }
    // Thread 0 writes total count
    if (tid == n - 1) {
        *out_count = prefix_sum[tid] + mask[tid];
    }
}

// ------------------------------------------------------------------
// Interleaved int/float memory pattern.

__global__ void interleaved_io(int *iout, float *fout, int *iin, float *fin, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int   iv = iin[tid];
        float fv = fin[tid];
        iout[tid] = iv + (int)fv;
        fout[tid] = (float)iv + fv;
    }
}

// ------------------------------------------------------------------
// Double-precision reduction.

__global__ void dbl_reduce(double *out, double *in, int n) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}
