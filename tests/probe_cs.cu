// Probe: Unusual parameter types and calling conventions
// - long long parameter
// - double** double pointer
// - const struct* parameter
// - void* parameter (cast in body)
// - Multiple qualifier combinations

__device__ long long multiply_ll(long long a, long long b) {
    return a * b;
}

__global__ void longlong_kernel(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        out[tid] = multiply_ll(v, v);
    }
}

__global__ void uint_kernel(unsigned int *out, unsigned int *in, int n) {
    unsigned int tid = (unsigned int)threadIdx.x;
    if ((int)tid < n) {
        unsigned int v = in[tid];
        // Rotating shift
        unsigned int lo = v >> 16;
        unsigned int hi = v << 16;
        out[tid] = hi | lo;
    }
}

// Double precision kernel
__global__ void double_kernel(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double r = v * v + v + 1.0;
        out[tid] = r;
    }
}
