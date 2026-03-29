// Probe: __constant__ memory, printf, typedef complex patterns,
// and device-to-device struct passing.

// ------------------------------------------------------------------
// __constant__ memory read in kernel.

__constant__ float c_coeffs[8];
__constant__ int   c_n;

__global__ void const_mem_fir(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int k = 0; k < 8; k++) {
            int src = tid - k;
            if (src >= 0) {
                acc += c_coeffs[k] * in[src];
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// __constant__ scalar used as loop bound.

__global__ void const_bound_loop(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid < c_n) {
        int acc = 0;
        for (int i = 0; i < c_n; i++) {
            if (in[i] > 0) acc += in[i];
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// typedef for scalar type alias.

typedef unsigned int uint32;
typedef long long    int64;

__global__ void typedef_arith(uint32 *out, uint32 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint32 v = in[tid];
        uint32 r = (v * 1234567u) ^ (v >> 16);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// typedef struct (common pattern).

typedef struct {
    float real;
    float imag;
} Complex;

__device__ Complex complex_mul(Complex a, Complex b) {
    Complex r;
    r.real = a.real * b.real - a.imag * b.imag;
    r.imag = a.real * b.imag + a.imag * b.real;
    return r;
}

__global__ void complex_mult_kernel(float *out_r, float *out_i,
                                     float *in_r, float *in_i, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Complex a, b;
        a.real = in_r[tid * 2];     a.imag = in_i[tid * 2];
        b.real = in_r[tid * 2 + 1]; b.imag = in_i[tid * 2 + 1];
        Complex c = complex_mul(a, b);
        out_r[tid] = c.real;
        out_i[tid] = c.imag;
    }
}

// ------------------------------------------------------------------
// printf in device code.

__global__ void debug_print(int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid == 0) {
        printf("Thread 0: in[0] = %d\n", in[0]);
    }
}

// ------------------------------------------------------------------
// Conditional printf (only some threads print).

__global__ void cond_print(float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        if (v > 100.0f) {
            printf("tid=%d val=%.2f exceeds threshold\n", tid, v);
        }
    }
}
