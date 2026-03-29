// Probe: __launch_bounds__ variations, kernel with many register hints,
// dynamic parallelism stubs, and kernel parameter edge cases.

// ------------------------------------------------------------------
// __launch_bounds__ with single arg.

__global__ void __launch_bounds__(256) lb_256(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = tid;
}

// ------------------------------------------------------------------
// __launch_bounds__ with two args.

__global__ void __launch_bounds__(128, 4) lb_128_4(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = in[tid] * 2;
}

// ------------------------------------------------------------------
// Kernel with many parameters (register passing stress).

__global__ void many_params(int *out,
                             int a, int b, int c, int d,
                             float e, float f, float g,
                             int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a + b + c + d + (int)(e + f + g);
    }
}

// ------------------------------------------------------------------
// Kernel with double parameters.

__global__ void double_params(double *out,
                               double a, double b, double c,
                               int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a * (double)tid + b + c;
    }
}

// ------------------------------------------------------------------
// Kernel with mixed pointer and scalar params.

__global__ void mixed_params(float * __restrict__ out,
                              const float * __restrict__ in,
                              float scale, float offset,
                              int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = tid * stride;
        out[idx] = in[idx] * scale + offset;
    }
}

// ------------------------------------------------------------------
// Empty kernel (just returns).

__global__ void empty_kernel() {
    // Nothing — tests that empty kernels emit correctly.
}

// ------------------------------------------------------------------
// Kernel using only blockIdx (no threadIdx arithmetic).

__global__ void block_only(int *out, int val) {
    int bid = blockIdx.x;
    out[bid] = val * bid;
}

// ------------------------------------------------------------------
// Kernel that writes to multiple distinct global arrays.

__global__ void multi_output(int *a, int *b, int *c, int *d, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        a[tid] = v;
        b[tid] = v * 2;
        c[tid] = v + 10;
        d[tid] = v ^ 0xFF;
    }
}
