// Probe: struct brace-initializer, array brace-initializer, __constant__ memory,
// __device__ global variable, multi-dim local array, designated field update,
// nested for with continue, prefix-sum on local array, complex condition chains,
// and __float2int_rn/__int2float_rn explicit conversions.

// ------------------------------------------------------------------
// __constant__ memory array (read by all threads).

__constant__ float cweights[8] = {0.1f, 0.15f, 0.15f, 0.1f,
                                    0.1f, 0.15f, 0.15f, 0.1f};
__constant__ int clut[4] = {10, 20, 30, 40};

__global__ void const_read(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        for (int k = 0; k < 8; k++) s += in[tid] * cweights[k];
        out[tid] = s;
    }
}

__global__ void const_lut(int *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 3;
        out[tid] = clut[i];
    }
}

// ------------------------------------------------------------------
// __device__ global (single int counter, for demonstration).

__device__ int g_counter = 0;

__global__ void device_global_inc(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(&g_counter, 1);
        out[tid] = g_counter;
    }
}

// ------------------------------------------------------------------
// Struct brace-initializer.

struct RGB { unsigned char r, g, b, pad; };

__device__ struct RGB make_rgb(unsigned char r, unsigned char g, unsigned char b) {
    struct RGB c = {r, g, b, 0};
    return c;
}

__global__ void rgb_kernel(int *out, unsigned char *r, unsigned char *g,
                             unsigned char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct RGB c = make_rgb(r[tid], g[tid], b[tid]);
        // Pack into int
        out[tid] = ((int)c.r << 16) | ((int)c.g << 8) | (int)c.b;
    }
}

// ------------------------------------------------------------------
// Array brace-initializer (local fixed-size LUT).

__global__ void local_lut(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lut[8] = {0, 1, 4, 9, 16, 25, 36, 49};
        int v = in[tid] & 7;
        out[tid] = lut[v];
    }
}

// ------------------------------------------------------------------
// Multi-dim local array (4x4 LUT).

__global__ void local_2d_lut(int *out, int *row_in, int *col_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mat[4][4] = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };
        int r = row_in[tid] & 3;
        int c = col_in[tid] & 3;
        out[tid] = mat[r][c];
    }
}

// ------------------------------------------------------------------
// Nested for with continue (skip diagonal elements).

__global__ void off_diag_sum(int *out, int *mat, int n, int dim) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int r = 0; r < dim; r++) {
            for (int c = 0; c < dim; c++) {
                if (r == c) continue;
                s += mat[r * dim + c];
            }
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Prefix-sum on local array, then output by index.

__global__ void local_prefix(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a[8];
        for (int i = 0; i < 8; i++) a[i] = in[(tid * 8 + i) % n];
        // prefix sum in-place
        for (int i = 1; i < 8; i++) a[i] += a[i-1];
        out[tid] = a[7];  // last element = total sum
    }
}

// ------------------------------------------------------------------
// __float2int_rn / __float2int_rz / __float2int_ru / __float2int_rd

__global__ void float2int_modes(int *out_rn, int *out_rz, int *out_ru, int *out_rd,
                                   float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_rn[tid] = __float2int_rn(in[tid]);
        out_rz[tid] = __float2int_rz(in[tid]);
        out_ru[tid] = __float2int_ru(in[tid]);
        out_rd[tid] = __float2int_rd(in[tid]);
    }
}

// ------------------------------------------------------------------
// __int2float_rn / __float2uint_rn.

__global__ void int2float_cvt(float *out_f, unsigned *out_u,
                                int *in_i, float *in_f, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_f[tid] = __int2float_rn(in_i[tid]);
        out_u[tid] = __float2uint_rn(in_f[tid]);
    }
}

// ------------------------------------------------------------------
// Complex condition chain: multi-condition &&/|| with side effects in ternary.

__global__ void complex_cond(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid], z = c[tid];
        // Multi-condition expression
        int r = (x > 0 && y > 0 && z > 0) ? x + y + z :
                (x < 0 || y < 0 || z < 0) ? x - y - z : 0;
        out[tid] = r;
    }
}
