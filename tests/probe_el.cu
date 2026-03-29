// Probe: nested struct pointer chains — a->b.field, a->b->field
// (a is ptr-to-struct containing ptr-to-struct)

struct Inner {
    float val;
    int count;
};

struct Outer {
    Inner *inner;
    int id;
};

// Also: struct returned by device function, then field accessed
__device__ Inner make_inner(float v, int c) {
    Inner r;
    r.val = v;
    r.count = c;
    return r;
}

__global__ void nested_ptr(float *out, Inner *inners, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Inner *p = &inners[tid];
        out[tid] = p->val * (float)p->count;
    }
}

__global__ void device_return_field(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Inner tmp = make_inner(in[tid], tid + 1);
        out[tid] = tmp.val * (float)tmp.count;
    }
}
