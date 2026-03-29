// Probe: complex C++ inheritance-style patterns that commonly appear in CUDA code
// — base class with virtual methods ignored, interface-style structs

// Interface-like struct (CUDA doesn't have vtables but code may be C++ style)
struct IKernel {
    int id;
    float scale;
};

struct ConvKernel {
    int id;
    float scale;
    int kw, kh;
    float *weights;
};

__device__ float apply_kernel(IKernel *k, float v) {
    return v * k->scale + (float)k->id;
}

__global__ void kernel_apply(float *out, float *in, IKernel *kernels, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        IKernel *k = &kernels[tid % 4];
        out[tid] = apply_kernel(k, in[tid]);
    }
}

// Struct with function pointer — parse without error (ptr is just a field)
struct Dispatch {
    int type;
    int data;
};

__device__ float dispatch_compute(Dispatch d, float v) {
    if (d.type == 0) return v * (float)d.data;
    if (d.type == 1) return v + (float)d.data;
    return v;
}

__global__ void dispatch_kernel(float *out, float *in, Dispatch *dispatches, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dispatch_compute(dispatches[tid % 4], in[tid]);
    }
}
