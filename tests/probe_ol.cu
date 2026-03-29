// Probe: array of structs (AoS), shared memory with struct types,
// atomic operations on shared memory, and cooperative patterns.

// ------------------------------------------------------------------
// Array of structs: data[i].x and data[i].y access.
// Each element is a struct; index computation is by struct size.

struct Pt2 { float x; float y; };

__global__ void aos_access(float *out, Pt2 *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // data[tid].x and data[tid].y — struct is 8 bytes
        float dx = data[tid].x;
        float dy = data[tid].y;
        out[tid] = dx * dx + dy * dy;  // squared distance from origin
    }
}

// ------------------------------------------------------------------
// Array of structs with write: data[i].x = ... ; data[i].y = ...

__global__ void aos_write(Pt2 *data, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        data[tid].x = xs[tid];
        data[tid].y = ys[tid];
    }
}

// ------------------------------------------------------------------
// Shared memory array of structs: each thread writes its own Pt2,
// then reads neighbor's value.

struct Pt2i { int x; int y; };

__global__ void shared_aos(int *out, int n) {
    __shared__ Pt2i sdata[256];
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    if (tid < n && tid < bsz) {
        sdata[tid].x = tid;
        sdata[tid].y = tid * 2;
        __syncthreads();
        int next = (tid + 1) % bsz;
        out[tid] = sdata[next].x + sdata[next].y;
    }
}

// ------------------------------------------------------------------
// Atomic operations: atomicAdd on int and float in global memory.

__global__ void atomic_ops(int *int_out, float *float_out, int *data,
                            float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(int_out, data[tid]);
        atomicAdd(float_out, fdata[tid]);
    }
}

// ------------------------------------------------------------------
// Warp reduction using __shfl_down_sync: sum across 32 lanes.

__global__ void warp_reduce(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = data[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if ((tid & 31) == 0) {
            out[tid >> 5] = val;
        }
    }
}
