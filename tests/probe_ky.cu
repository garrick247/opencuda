// Probe: unsigned comparisons use setp.u32 not setp.s32,
// mixed pointer widths (short/byte element sizes),
// multi-dimensional grid (blockIdx.y / gridDim.x),
// atomicCAS and atomicExch

// Unsigned comparison: tid < n where both are unsigned
__global__ void unsigned_tid_bound(unsigned int *out, unsigned int *in,
                                    unsigned int n) {
    unsigned int tid = (unsigned int)threadIdx.x;
    if (tid < n) {      // should use setp.lt.u32
        out[tid] = in[tid] + 1u;
    }
}

// Byte-element array (char): stride 1
__global__ void byte_array(unsigned char *out, unsigned char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char v = in[tid];
        out[tid] = v + (unsigned char)1;
    }
}

// 2D grid: row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x
__global__ void grid2d_index(int *out, int *in, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = row * width + col;
        out[idx] = in[idx] * 2;
    }
}

// atomicAdd and atomicCAS
__global__ void atomic_ops(int *counter, int *lock, int val) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // atomicAdd: returns old value
        int old = atomicAdd(counter, val);
        // atomicCAS: CAS with expected=0, desired=1
        int prev = atomicCAS(lock, 0, 1);
        // atomicExch
        int swapped = atomicExch(counter, 42);
        // Store results
        counter[1] = old;
        counter[2] = prev;
        counter[3] = swapped;
    }
}

// atomicMin/Max on global memory
__global__ void atomic_minmax(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicMin(&out[0], in[tid]);
        atomicMax(&out[1], in[tid]);
    }
}
