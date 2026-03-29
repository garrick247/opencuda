// Probe: complex macro token patterns, multi-dimensional global arrays,
// accumulated writes to multiple global locations, and arithmetic
// precision patterns.

// ------------------------------------------------------------------
// Token-pasting macro (## operator) for name mangling.

#define MAKE_KERNEL(suffix) kernel_##suffix
#define MAKE_FN(suffix) device_fn_##suffix

__device__ int MAKE_FN(add)(int a, int b) { return a + b; }
__device__ int MAKE_FN(mul)(int a, int b) { return a * b; }

__global__ void MAKE_KERNEL(add)(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = MAKE_FN(add)(a[tid], b[tid]);
}

__global__ void MAKE_KERNEL(mul)(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = MAKE_FN(mul)(a[tid], b[tid]);
}

// ------------------------------------------------------------------
// Macro with numeric constant.

#define VERSION 42
#define VERSION_MAJOR 1
#define VERSION_MINOR 5

__global__ void macro_version(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = VERSION * 1000 + VERSION_MAJOR * 10 + VERSION_MINOR + tid;
        // 42*1000 + 10 + 5 + tid = 42015 + tid
    }
}

// ------------------------------------------------------------------
// Flat global array used as logical 2D (row-major).

#define ROWS 4
#define COLS 8
__device__ int g_matrix[ROWS * COLS];

__global__ void write_matrix(int n) {
    int tid = threadIdx.x;
    if (tid < ROWS * COLS) {
        int r = tid / COLS;
        int c = tid % COLS;
        g_matrix[r * COLS + c] = r * COLS + c;
    }
}

__global__ void read_matrix(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < ROWS * COLS) {
        int r = tid / COLS;
        int c = tid % COLS;
        out[tid] = g_matrix[r * COLS + c];
    }
}

// ------------------------------------------------------------------
// Accumulated writes: each thread writes to multiple locations.

__global__ void scatter_write(int *out, int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Write to 4 derived locations
        out[v % n]           = tid;
        out[(v * 2 + 1) % n] = tid * 2;
        out[(v * 3 + 2) % n] = tid * 3;
        out[(v * 5 + 3) % n] = tid * 5;
    }
}

// ------------------------------------------------------------------
// Long arithmetic chain: test that optimizer doesn't over-fold.

__global__ void long_arith_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Chain of 16 operations that depend on each other
        v = v * 3 + 1;
        v = (v ^ (v >> 4));
        v = v * 0x85EBCA6B;
        v = (v ^ (v >> 13));
        v = v * 0xC2B2AE35;
        v = (v ^ (v >> 16));
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Double precision sin/cos (via float approximation).

__global__ void dbl_sincos(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        // sin^2 + cos^2 = 1.0 (use float approx)
        float fv = (float)v;
        float s = sinf(fv);
        float c = cosf(fv);
        out[tid] = (double)(s * s + c * c);  // should be ≈ 1.0
    }
}

// ------------------------------------------------------------------
// Vector normalization (3D).

__global__ void normalize3d(float *ox, float *oy, float *oz,
                              float *ix, float *iy, float *iz, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = ix[tid];
        float y = iy[tid];
        float z = iz[tid];
        float len_sq = x*x + y*y + z*z;
        float inv_len = (len_sq > 0.0f) ? rsqrtf(len_sq) : 0.0f;
        ox[tid] = x * inv_len;
        oy[tid] = y * inv_len;
        oz[tid] = z * inv_len;
    }
}

// ------------------------------------------------------------------
// Conditional pointer: compute address from one of two arrays.

__device__ float *select_array(float *a, float *b, int cond) {
    return cond ? a : b;
}

__global__ void ptr_select_fn(float *out, float *a, float *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *src = select_array(a, b, sel[tid]);
        out[tid] = src[tid];
    }
}

// ------------------------------------------------------------------
// Compute CRC-like checksum via bit operations.

__global__ void crc_like(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int crc = 0xFFFFFFFF;
        unsigned int v = in[tid];
        // Process 8 "bits" of v
        for (int i = 0; i < 8; i++) {
            unsigned int bit = (v ^ crc) & 1;
            crc >>= 1;
            if (bit) crc ^= 0xEDB88320;
            v >>= 1;
        }
        out[tid] = crc ^ 0xFFFFFFFF;
    }
}
