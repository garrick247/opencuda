// Probe: many kernels + device functions in one TU — tests
// module-level state, symbol resolution, and PTX .visible/.entry ordering.

// A bank of 8 device utility functions.
__device__ int clz32(unsigned int v) { return __clz(v); }
__device__ int popc32(unsigned int v) { return __popc(v); }
__device__ unsigned int rev32(unsigned int v) { return __brev(v); }
__device__ int min3(int a, int b, int c) { return min(min(a, b), c); }
__device__ int max3(int a, int b, int c) { return max(max(a, b), c); }
__device__ float lerp_f(float a, float b, float t) { return a + t * (b - a); }
__device__ float sq_f(float x) { return x * x; }
__device__ float hypot2(float x, float y) { return sq_f(x) + sq_f(y); }

// 8 kernels, each using a different subset of device functions.

__global__ void k0(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = (unsigned int)clz32(in[tid]);
}

__global__ void k1(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = (unsigned int)popc32(in[tid]);
}

__global__ void k2(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = rev32(in[tid]);
}

__global__ void k3(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = min3(a[tid], b[tid], c[tid]);
}

__global__ void k4(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = max3(a[tid], b[tid], c[tid]);
}

__global__ void k5(float *out, float *a, float *b, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = lerp_f(a[tid], b[tid], t[tid]);
}

__global__ void k6(float *out, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = hypot2(x[tid], y[tid]);
}

__global__ void k7(float *out, float *x, float *y, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float h = hypot2(x[tid], y[tid]);
        float l = lerp_f(x[tid], y[tid], t[tid]);
        out[tid] = h + l;
    }
}

// Additional kernels using multiple device functions together.
__global__ void k8(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid] = (unsigned int)(clz32(v) + popc32(v)) + rev32(v);
    }
}

__global__ void k9(int *out, int *a, int *b, int *c, float *f, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mn = min3(a[tid], b[tid], c[tid]);
        int mx = max3(a[tid], b[tid], c[tid]);
        float lp = lerp_f((float)mn, (float)mx, f[tid]);
        out[tid] = (int)lp;
    }
}
