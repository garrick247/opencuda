// Atomic operations for runtime validation.
// Signature: (int *out, int *a, int *b, int n) — a is input, out is atomic target.

// Atomic sum: each thread atomicAdd its value
__global__ void atomic_sum(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicAdd(out, a[gid]);
}

// Atomic max: find the maximum of all elements
__global__ void atomic_max_k(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicMax(out, a[gid]);
}

// Atomic min: find the minimum of all elements
__global__ void atomic_min_k(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicMin(out, a[gid]);
}

// Atomic OR: bitwise OR of all elements
__global__ void atomic_or_k(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicOr(out, a[gid]);
}

// Ballot-based population count: count elements > 0
__global__ void ballot_count(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(0xFFFFFFFF, gid < n && a[gid] > 0);
    int lane = threadIdx.x & 31;
    if (lane == 0) atomicAdd(out, __popc(mask));
}
