// Probe: CUDA intrinsics used in expressions (not just standalone),
// threadIdx/blockIdx in complex formulas, and intrinsics as arguments.

// ------------------------------------------------------------------
// threadIdx.x used in nested arithmetic expression.

__global__ void tid_expr(int *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Complex formula using tid directly
        int r = (tid * tid + tid * 3 + 7) % n;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// threadIdx as array subscript (indirect).

__global__ void tid_indirect(int *out, int *lut, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int slot = lut[tid % 16];
        out[tid] = slot * (int)threadIdx.x;
    }
}

// ------------------------------------------------------------------
// threadIdx.y and threadIdx.z used.

__global__ void tid_3d(int *out, int nx, int ny) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int idx = tz * ny * nx + ty * nx + tx;
    out[idx] = tx + ty * 10 + tz * 100;
}

// ------------------------------------------------------------------
// Intrinsic result passed directly to function.

__device__ int triple(int x) { return x * 3; }

__global__ void intrinsic_as_arg(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = triple((int)threadIdx.x + (int)blockIdx.x);
    }
}

// ------------------------------------------------------------------
// warpSize used in computation.

__global__ void warpsize_use(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lane = tid % warpSize;
        int warp = tid / warpSize;
        out[tid] = lane + warp * warpSize;
    }
}

// ------------------------------------------------------------------
// Atomic result used in subsequent computation.

__global__ void atomic_result_use(int *out, int *counter, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && in[tid] > 0) {
        int slot = atomicAdd(counter, 1);
        out[slot] = in[tid];
    }
}

// ------------------------------------------------------------------
// __syncthreads in multiple positions within a kernel.

__global__ void multi_sync(float *out, float *in, int n) {
    __shared__ float buf[256];
    int tid = threadIdx.x;
    // Phase 1
    buf[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    // Phase 2: each thread reads its neighbor
    float v = buf[tid];
    float left = (tid > 0) ? buf[tid - 1] : 0.0f;
    __syncthreads();
    // Phase 3: write result back
    buf[tid] = v + left;
    __syncthreads();
    if (tid < n) out[tid] = buf[tid];
}

// ------------------------------------------------------------------
// blockDim/gridDim in conditional.

__global__ void block_cond(int *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int is_first_block = (blockIdx.x == 0) ? 1 : 0;
        int is_last_thread = (threadIdx.x == blockDim.x - 1) ? 1 : 0;
        out[tid] = is_first_block * 10 + is_last_thread;
    }
}

// ------------------------------------------------------------------
// Nested atomics: atomicAdd of atomicCAS result (unusual but legal).

__global__ void nested_atomic(int *out, int *lock, int *counter, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Spin-acquire a simple lock (single attempt, not real spin)
        int old = atomicCAS(lock, 0, 1);
        if (old == 0) {
            // We hold the lock
            int v = atomicAdd(counter, 1);
            out[tid] = v;
            atomicExch(lock, 0);  // release
        } else {
            out[tid] = -1;  // didn't acquire
        }
    }
}
