// Probe: typedef struct, atomicCAS/atomicExch, 2D/3D thread indexing,
// __shared__ dynamic index, atomicMin/atomicMax, and
// multi-kernel file with shared typedef.

// ------------------------------------------------------------------
// typedef struct.

typedef struct {
    int   id;
    float score;
} Record;

__global__ void typedef_struct(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Record r;
        r.id    = tid;
        r.score = in[tid] * (float)tid;
        out[tid] = r.score;
    }
}

// ------------------------------------------------------------------
// 2D thread/block indexing: 2D kernel addressing.

__global__ void kernel_2d(float *out, float *in, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = in[idx] * 2.0f;
    }
}

// ------------------------------------------------------------------
// 3D thread indexing.

__global__ void kernel_3d(float *out, float *in,
                           int dx, int dy, int dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dx && y < dy && z < dz) {
        int idx = z * dy * dx + y * dx + x;
        out[idx] = in[idx] + 1.0f;
    }
}

// ------------------------------------------------------------------
// atomicCAS: compare-and-swap.

__global__ void atomic_cas(int *out, int *lock, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // CAS: if lock[0] == 0, set to 1
        int old = atomicCAS(lock, 0, 1);
        out[tid] = old;  // 0 for the winner, 1 for all others
    }
}

// ------------------------------------------------------------------
// atomicExch: atomic exchange.

__global__ void atomic_exch(int *out, int *slot, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int old = atomicExch(slot, tid);
        out[tid] = old;
    }
}

// ------------------------------------------------------------------
// atomicMin / atomicMax on global array.

__global__ void atomic_min_max(int *gmin, int *gmax, int *in, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        atomicMin(gmin, in[gid]);
        atomicMax(gmax, in[gid]);
    }
}

// ------------------------------------------------------------------
// __shared__ with runtime index — forces dynamic indexing via st/ld.shared.

__global__ void shared_dynamic_idx(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        smem[tid] = in[gid];
    } else {
        smem[tid] = 0;
    }
    __syncthreads();

    // Dynamic index: read from neighbor at runtime-computed offset
    int neighbor = (tid + in[gid < n ? gid : 0] % blockDim.x) % blockDim.x;
    int v = smem[neighbor];
    if (gid < n) {
        out[gid] = v;
    }
}

// ------------------------------------------------------------------
// atomicAdd on float (CUDA 2.0+).

__global__ void atomic_add_float(float *sum, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicAdd(sum, in[gid]);
    }
}

// ------------------------------------------------------------------
// gridDim usage.

__global__ void uses_griddim(int *out, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    if (tid < n) {
        out[tid] = tid + total;
    }
}
