// Probe: PtrTy field in struct with different caller/param names,
// and other edge cases the v0.64 fixes introduced.

// ------------------------------------------------------------------
// Struct with pointer field, caller variable and param have DIFFERENT names.
// Previously would silently use caller's `buf_ptr` for `b.ptr` — now fixed.

struct Slice {
    int *ptr;
    int  len;
};

__device__ int slice_sum(Slice b, int start) {
    int s = 0;
    for (int i = start; i < b.len; i++) {
        s += b.ptr[i];
    }
    return s;
}

__global__ void renamed_struct_field(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Slice buf;           // named "buf", param is "b"
        buf.ptr = data;
        buf.len = n;
        out[0] = slice_sum(buf, 0);
    }
}

// ------------------------------------------------------------------
// Struct from array element with pointer field.
// Tests the load-from-pointer path for PtrTy fields.
// (Pass a struct-array element by value to a device fn.)

struct Header {
    int  count;
    int *data;
};

__device__ int header_sum(Header h) {
    int s = 0;
    for (int i = 0; i < h.count; i++) {
        s += h.data[i];
    }
    return s;
}

__global__ void ptr_field_from_array(int *out, Header *headers, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // headers[tid] has a pointer field — tests PtrTy load in _load_param_fields
        out[tid] = header_sum(headers[tid]);
    }
}

// ------------------------------------------------------------------
// Two device functions with same param name for different struct types.
// Tests that struct field isolation works when the same param name is reused.

struct P2 { float x, y; };
struct P3 { float x, y, z; };

__device__ float len2(P2 p) { return p.x * p.x + p.y * p.y; }
__device__ float len3(P3 p) { return p.x * p.x + p.y * p.y + p.z * p.z; }

__global__ void two_struct_types(float *out, P2 *p2s, P3 *p3s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = len2(p2s[tid]);
        float b = len3(p3s[tid]);
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Struct passed twice to same device fn — both calls must load correctly.

__device__ float dot2f(P2 a, P2 b) {
    return a.x * b.x + a.y * b.y;
}

__global__ void double_struct_call(float *out, P2 *ps, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Call dot2f twice with different array elements
        float r = dot2f(ps[tid], ps[(tid + 1) % n]);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Struct with pointer + scalar, both used in device fn, assigned before call.

struct Work {
    float *vals;
    int    count;
    float  scale;
};

__device__ float work_apply(Work w) {
    float s = 0.0f;
    for (int i = 0; i < w.count; i++) {
        s += w.vals[i] * w.scale;
    }
    return s;
}

__global__ void mixed_struct_fields(float *out, float *vals, int n, float scale) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Work w;
        w.vals  = vals;
        w.count = n;
        w.scale = scale;
        out[0] = work_apply(w);
    }
}
