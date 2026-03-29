// Probe: long long arithmetic, multiple kernels sharing device fns,
// shared struct arrays, warp vote in conditions, pointer difference.

// ------------------------------------------------------------------
// Long long (64-bit) integer arithmetic.

__global__ void ll_arithmetic(long long *out, long long *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long va = a[tid], vb = b[tid];
        out[tid] = va * vb + va - vb;
    }
}

// ------------------------------------------------------------------
// Unsigned long long (64-bit) arithmetic.

__global__ void ull_arithmetic(unsigned long long *out,
                                unsigned long long *a, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = a[tid];
        out[tid] = v * v + 1ULL;
    }
}

// ------------------------------------------------------------------
// Long long comparison and conditional.

__global__ void ll_compare(int *out, long long *data, long long threshold, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (data[tid] > threshold) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Multiple kernels sharing the same __device__ function.
// Both kernel_a and kernel_b call shared_fn — tests that inlining
// into each kernel produces correct, independent code.

__device__ int shared_fn(int x, int y) {
    return (x > y) ? x - y : y - x;
}

__global__ void kernel_a(int *out, int *x, int *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = shared_fn(x[tid], y[tid]);
    }
}

__global__ void kernel_b(int *out, int *x, int *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Different usage pattern of the same device function
        out[tid] = shared_fn(x[tid], 0) + shared_fn(0, y[tid]);
    }
}

// ------------------------------------------------------------------
// __shared__ array of a simple struct.
// Tests that shared memory works with struct-typed elements.

struct Pair {
    int key, val;
};

__global__ void shared_struct_array(int *out, int *keys, int *vals, int n) {
    __shared__ Pair smem[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        smem[tid].key = keys[tid];
        smem[tid].val = vals[tid];
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        // Simple reduction: sum all values where key > 0
        int sum = 0;
        for (int i = 0; i < n && i < 32; i++) {
            if (smem[i].key > 0) {
                sum += smem[i].val;
            }
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Warp vote __any_sync in if condition.

__global__ void warp_any_cond(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? data[tid] : 0;
    // Check if any thread in the warp has a positive value
    if (__any_sync(0xFFFFFFFF, v > 0)) {
        out[tid] = v + 1;
    } else {
        out[tid] = 0;
    }
}

// ------------------------------------------------------------------
// Warp vote __all_sync in loop condition.

__global__ void warp_all_loop(int *out, int *data, int n, int rounds) {
    int tid = threadIdx.x;
    int v = (tid < n) ? data[tid] : 1;
    int r = 0;
    for (int i = 0; i < rounds && __all_sync(0xFFFFFFFF, v > 0); i++) {
        v = v - 1;
        r++;
    }
    out[tid] = r;
}
