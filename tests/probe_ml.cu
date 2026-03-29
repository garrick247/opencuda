// Probe: Struct store patterns and pointer-to-struct edge cases
// - Array-of-structs write: arr[i] = computed_struct
// - Struct output parameter: write struct through pointer
// - Conditional struct store (different branch stores)
// - Struct store in loop
// - Device function taking struct* and modifying it

struct Pair {
    float a, b;
};

struct Triple {
    float x, y, z;
};

// Array-of-structs write via subscript
__global__ void aos_write(Pair *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair p;
        p.a = xs[tid];
        p.b = ys[tid];
        out[tid] = p;
    }
}

// Output via pointer parameter
__device__ void make_triple(float x, float y, float z, Triple *out) {
    out->x = x;
    out->y = y;
    out->z = z;
}

__global__ void triple_via_ptr(Triple *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        make_triple(in[tid], in[tid]*2.0f, in[tid]*3.0f, &out[tid]);
    }
}

// Conditional struct store: different Pair written in each branch
__global__ void cond_struct_store(Pair *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        Pair p;
        if (v > 0.0f) {
            p.a = v;
            p.b = v * 2.0f;
        } else {
            p.a = -v;
            p.b = 0.0f;
        }
        out[tid] = p;
    }
}

// Struct store in loop accumulator pattern
__global__ void loop_struct_store(Pair *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            Pair p;
            p.a = xs[i];
            p.b = ys[i];
            out[i] = p;
        }
    }
}

// Return struct from device and immediately store it
__device__ Pair make_pair(float x, float y) {
    Pair p;
    p.a = x;
    p.b = y;
    return p;
}

__global__ void inline_and_store(Pair *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = make_pair(xs[tid], ys[tid]);
    }
}

// Struct field stored in nested subscript
__global__ void nested_struct_store(Triple *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Triple t;
        float v = in[tid];
        t.x = v;
        t.y = v * v;
        t.z = v * v * v;
        out[tid] = t;
    }
}
