// Probe: multiple calls to the same device function in one kernel,
// device function called in both branches of an if,
// chained device function results

__device__ int square(int x) { return x * x; }
__device__ int cube(int x) { return x * x * x; }
__device__ int clamp_s(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Two calls to square() in same kernel
__global__ void two_squares(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = square(in[tid]);
        int b = square(in[tid] + 1);
        out[tid] = a + b;
    }
}

// square() and cube() called in same kernel
__global__ void sq_plus_cube(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = square(in[tid]) + cube(in[tid]);
    }
}

// Device function called in if-true and if-false branches
__global__ void branch_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        if (v > 0) {
            result = square(v);
        } else {
            result = cube(-v);
        }
        out[tid] = result;
    }
}

// Chained: result of one call fed into another
__global__ void chained_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // clamp(square(x), 0, 100)
        out[tid] = clamp_s(square(in[tid]), 0, 100);
    }
}
