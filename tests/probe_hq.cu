// Probe: struct with pointer fields in local memory (auto var),
// storing/loading pointer-typed fields from local struct

struct PtrPair {
    float *a;
    float *b;
};

__global__ void local_struct_ptrs(float *out, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        PtrPair pp;
        pp.a = x;
        pp.b = y;
        out[tid] = pp.a[tid] + pp.b[tid];
    }
}

// Struct with mixed pointer and int fields
struct Slice {
    float *data;
    int len;
    int offset;
};

__global__ void slice_sum(float *out, float *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Slice s;
        s.data = buf;
        s.len = n;
        s.offset = tid;
        float val = s.data[s.offset];
        out[tid] = val;
    }
}
