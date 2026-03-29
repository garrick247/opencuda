// Probe: Edge cases in struct field access chains
// - Struct field that's a struct (nested)
// - Array element that's a struct, then field access
// - Struct pointer, then arrow, then field of nested struct
// - Mixed dot and arrow chains

struct Inner {
    float val;
    int count;
};

struct Outer {
    Inner a;
    Inner b;
    float weight;
};

__device__ float process_outer(Outer o) {
    return o.a.val * o.weight + o.b.val * (1.0f - o.weight);
}

__device__ float process_outer_ptr(Outer *o) {
    return o->a.val * o->weight + o->b.val * (1.0f - o->weight);
}

__global__ void nested_struct_kernel(float *out, Outer *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Outer o = data[tid];
        float r1 = process_outer(o);
        float r2 = process_outer_ptr(&data[tid]);
        out[tid] = r1 + r2;
    }
}
