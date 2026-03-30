// Probe: nested switch, large switch (20 cases), device function calling
// device function with struct param, complex pointer arithmetic (stride
// access via typed pointer cast), global mem coalescing patterns,
// warp-level broadcast via __shfl_sync lane 0, mixed float/double
// accumulator, and __isnan/__isinf on float.

// ------------------------------------------------------------------
// Nested switch statement.

__device__ int nested_switch(int cat, int sub) {
    switch (cat) {
        case 0:
            switch (sub) {
                case 0: return 100;
                case 1: return 101;
                default: return 109;
            }
        case 1:
            switch (sub) {
                case 0: return 200;
                case 1: return 201;
                case 2: return 202;
                default: return 209;
            }
        case 2: return 300;
        default: return -1;
    }
}

__global__ void nested_switch_kernel(int *out, int *cats, int *subs, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = nested_switch(cats[tid], subs[tid]);
}

// ------------------------------------------------------------------
// Large switch (20 cases).

__device__ int classify20(int v) {
    switch (v % 20) {
        case  0: return 0;
        case  1: return 1;
        case  2: return 4;
        case  3: return 9;
        case  4: return 16;
        case  5: return 25;
        case  6: return 36;
        case  7: return 49;
        case  8: return 64;
        case  9: return 81;
        case 10: return 100;
        case 11: return 121;
        case 12: return 144;
        case 13: return 169;
        case 14: return 196;
        case 15: return 225;
        case 16: return 256;
        case 17: return 289;
        case 18: return 324;
        case 19: return 361;
        default: return -1;
    }
}

__global__ void classify20_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = classify20(in[tid]);
}

// ------------------------------------------------------------------
// Device function calling device function with struct param.

struct Transform { float scale; float offset; };

__device__ float apply_transform(struct Transform t, float x) {
    return t.scale * x + t.offset;
}

__device__ float chain_transforms(struct Transform t1, struct Transform t2, float x) {
    float y = apply_transform(t1, x);
    return apply_transform(t2, y);
}

__global__ void chain_xform_kernel(float *out, float *in,
                                      float s1, float o1, float s2, float o2, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Transform t1, t2;
        t1.scale = s1; t1.offset = o1;
        t2.scale = s2; t2.offset = o2;
        out[tid] = chain_transforms(t1, t2, in[tid]);
    }
}

// ------------------------------------------------------------------
// Warp broadcast: lane 0 broadcasts a value to all lanes.

__global__ void warp_broadcast(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int v = in[gid];
    // Broadcast lane 0's value to entire warp
    int b = __shfl_sync(0xFFFFFFFF, v, 0);
    out[gid] = v + b;
}

// ------------------------------------------------------------------
// Mixed float/double accumulator (float input, double accumulator).

__global__ void mixed_accum(double *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double s = 0.0;
        for (int i = 0; i < 16; i++) {
            s += (double)in[(tid * 16 + i) % n];
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// __isnanf / __isinff on float.

__global__ void float_special(int *out_nan, int *out_inf, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out_nan[tid] = __isnanf(v);
        out_inf[tid] = __isinff(v);
    }
}

// ------------------------------------------------------------------
// Complex stride access via typed pointer.

__global__ void stride_access(float *out, float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = in + tid * stride;
        float s = 0.0f;
        for (int i = 0; i < stride && i < 8; i++) {
            s += p[i];
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Kernel with no parameters except n (all computation is threadIdx).

__global__ void pure_compute(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Fibonacci-like sequence
        int a = 0, b = 1;
        for (int i = 0; i < (tid % 16); i++) {
            int tmp = a + b;
            a = b;
            b = tmp;
        }
        out[tid] = b;
    }
}

// ------------------------------------------------------------------
// Coalesced vs strided access pattern.

__global__ void coalesced_read(float *out, float *in, int n) {
    // Coalesced: thread i reads element i
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] * 2.0f;
}

__global__ void strided_read(float *out, float *in, int stride, int n) {
    // Strided: thread i reads element i*stride
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gid * stride;
    if (gid < n && idx < n) out[gid] = in[idx];
}
