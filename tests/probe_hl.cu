// Probe: complex initializer patterns — struct literal init,
// designated initializers (not standard in C++, but CUDA C is C-subset),
// multiple variables declared on one line with different initializers

struct Rect {
    float x, y, w, h;
};

__global__ void struct_init(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Struct initialized field-by-field (no literal, just assignments)
        Rect r;
        r.x = (float)tid;
        r.y = (float)(tid + 1);
        r.w = 2.0f;
        r.h = 3.0f;
        out[tid] = r.x + r.y + r.w + r.h;
    }
}

// Multiple vars on one declaration line
__global__ void multi_decl_init(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid], b = in[tid] * 2.0f, c = a + b;
        out[tid] = c;
    }
}

// Comma-separated declarations (int i, j, k)
__global__ void multi_int_decl(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = in[tid];
        b = a * 2;
        c = a + b;
        out[tid] = c;
    }
}
