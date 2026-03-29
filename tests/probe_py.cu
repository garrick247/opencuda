// Probe: printf patterns, volatile access, const __restrict__ pointers,
// and misc kernel patterns not yet covered.

// ------------------------------------------------------------------
// Printf in a loop (multiple calls).

__global__ void printf_loop(int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            printf("iter %d\n", i);
        }
    }
}

// ------------------------------------------------------------------
// Printf with mixed-type args.

__global__ void printf_mixed(int *data, float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && n > 0) {
        printf("int=%d float=%f\n", data[0], fdata[0]);
    }
}

// ------------------------------------------------------------------
// Printf after conditional.

__global__ void printf_cond(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (data[tid] < 0) {
            printf("neg at %d\n", tid);
        }
    }
}

// ------------------------------------------------------------------
// __restrict__ pointers: no aliasing, loads may be CSE'd.

__global__ void restrict_add(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b,
                              int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// const __restrict__ + loop: reduction with no aliasing.

__global__ void restrict_reduce(float *out,
                                 const float * __restrict__ data,
                                 int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Kernel with many parameters (> 8).
// Tests that all parameters are loaded correctly.

__global__ void many_params(int *out, int a, int b, int c, int d,
                             int e, int f, int g, int h) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = a + b + c + d + e + f + g + h;
    }
}

// ------------------------------------------------------------------
// Kernel with no output: side-effect only via atomics.

__device__ int g_side_count = 0;

__global__ void side_effect_only(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n && data[tid] > 0) {
        atomicAdd(&g_side_count, 1);
    }
}

// ------------------------------------------------------------------
// Double-precision sqrt and trig (via intrinsics available in PTX).

__global__ void double_math(double *out, double *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = data[tid];
        // Use only basic ops available without stdlib
        out[tid] = v * v + 1.0;
    }
}

// ------------------------------------------------------------------
// Kernel that uses blockIdx in all three dimensions.

__global__ void block3d(int *out, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int idx = z * height * width + y * width + x;
    if (x < width) {
        out[idx] = x + y * 10 + z * 100;
    }
}

// ------------------------------------------------------------------
// Kernel using gridDim.

__global__ void grid_dim_use(int *out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = gridDim.x * blockDim.x;
    out[tid % total] = tid;
}
