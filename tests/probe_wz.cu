// Probe: pragma unroll, launch_bounds variants, multi-dim thread indices,
// gridDim usage, complex indexing arithmetic, and register pressure via
// lots of independent float computations.

// ------------------------------------------------------------------
// #pragma unroll with explicit count.

__global__ void pragma_unroll(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < 8; i++) {
            sum += in[(tid + i) % n];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// __launch_bounds__ with single argument only.

__global__ __launch_bounds__(128) void lb_single(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 3;
    }
}

// ------------------------------------------------------------------
// 3D thread block indexing.

__global__ void kernel_3d(float *out, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        out[z * ny * nx + y * nx + x] = (float)(x + y * nx + z * nx * ny);
    }
}

// ------------------------------------------------------------------
// gridDim in all three dimensions.

__global__ void griddim_3d(int *out, int n) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int gx = gridDim.x;
    int gy = gridDim.y;
    // Flatten 3D grid index
    int idx = z * gy * gx + y * gx + x;
    if (idx < n) {
        out[idx] = idx;
    }
}

// ------------------------------------------------------------------
// Complex 3D index arithmetic.

__global__ void volume_copy(float *dst, float *src,
                             int Nx, int Ny, int Nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix < Nx && iy < Ny && iz < Nz) {
        int idx = iz * Ny * Nx + iy * Nx + ix;
        dst[idx] = src[idx];
    }
}

// ------------------------------------------------------------------
// Saturated addition: clamp sum to [0, 255].

__device__ unsigned char sat_add_u8(unsigned char a, unsigned char b) {
    unsigned int sum = (unsigned int)a + (unsigned int)b;
    return (unsigned char)(sum > 255 ? 255 : sum);
}

__global__ void sat_add_kernel(unsigned char *out, unsigned char *a,
                                unsigned char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sat_add_u8(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// RGB to grayscale: 0.299R + 0.587G + 0.114B.

__global__ void rgb_to_gray(unsigned char *gray, unsigned char *r,
                              unsigned char *g, unsigned char *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float rv = (float)r[tid];
        float gv = (float)g[tid];
        float bv = (float)b[tid];
        float lum = 0.299f * rv + 0.587f * gv + 0.114f * bv;
        gray[tid] = (unsigned char)lum;
    }
}

// ------------------------------------------------------------------
// Softmax-like: exp(x_i) / sum(exp(x_j)) via two passes.
// Pass 1: compute max (for numerical stability).

__global__ void find_max(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : -3.402823466e+38f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && smem[tid + stride] > smem[tid]) {
            smem[tid] = smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Element-wise operations: a variety of functions in a single pass.

__global__ void element_wise_zoo(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r;
        int mod4 = tid & 3;
        if (mod4 == 0) r = sinf(v) + cosf(v);
        else if (mod4 == 1) r = expf(v) * logf(fabsf(v) + 1.0f);
        else if (mod4 == 2) r = sqrtf(fabsf(v)) + rsqrtf(fabsf(v) + 1.0f);
        else r = floorf(v) + ceilf(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Integer power: i^k for small k via repeated multiply.

__device__ int ipow(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

__global__ void ipow_kernel(int *out, int *bases, int *exps, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ipow(bases[tid], exps[tid] & 7);  // cap at 7 to avoid overflow
    }
}
