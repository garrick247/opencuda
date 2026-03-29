// Probe: extern __shared__, __syncwarp, continue/break in device fn loops,
// bitwise operations on signed integers, max/min via conditional.

// ------------------------------------------------------------------
// extern __shared__ dynamic allocation: pointer typed as byte array,
// reinterpreted at kernel launch. Tests that the parameter is emitted
// as .extern .shared .align N .b8 name[]; in PTX.

__global__ void dyn_shared(float *out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    if (tid < n) {
        sdata[tid] = (float)tid;
        __syncthreads();
        out[tid] = sdata[n - 1 - tid];
    }
}

// ------------------------------------------------------------------
// __syncwarp(): must emit bar.warp.sync 0xffffffff; (or equivalent).
// Used for intra-warp synchronization without full __syncthreads.

__global__ void warp_sync_test(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    if (tid < n) {
        int v = data[tid];
        __syncwarp();
        out[tid] = v + lane;
    }
}

// ------------------------------------------------------------------
// Break/continue in a device function loop (inlined by the compiler).
// Tests that loop control flow inside inlined functions is correct.

__device__ int first_positive(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] > 0) return arr[i];
    }
    return -1;
}

__global__ void find_first(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = first_positive(data, n);
    }
}

// ------------------------------------------------------------------
// Bitwise NOT (~) on an integer value.
// Tests that ~x emits xor.b32 x, -1 (or not.b32 if available).

__global__ void bitwise_not(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ~data[tid];
    }
}

// ------------------------------------------------------------------
// Integer max/min using ternary (no intrinsic).
// Tests that the ternary-based max/min pattern generates correct setp+selp.

__global__ void clamp_to_range(int *out, int *data, int lo, int hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int clamped = (v < lo) ? lo : (v > hi) ? hi : v;
        out[tid] = clamped;
    }
}
