// Probe: complex struct operations — nested structs, struct array access,
// struct assignment, struct fields in loops, multi-level inline with structs.

// ------------------------------------------------------------------
// Struct with four fields: all scalar, accessed in loop.

struct Vec4 {
    float x, y, z, w;
};

__global__ void vec4_dot(float *out, Vec4 *a, Vec4 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec4 va = a[tid];
        Vec4 vb = b[tid];
        out[tid] = va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }
}

// ------------------------------------------------------------------
// Struct with mixed types: int index + float value.

struct IndexedVal {
    int   idx;
    float val;
};

__global__ void indexed_val_max(float *out, IndexedVal *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float best = data[0].val;
        for (int i = 1; i < n; i++) {
            if (data[i].val > best) best = data[i].val;
        }
        out[0] = best;
    }
}

// ------------------------------------------------------------------
// Nested struct: Matrix22 { Row2 r0, r1; }, Row2 { float a, b; }.
// Tests multi-level field access.

struct Row2 {
    float a, b;
};

struct Matrix22 {
    Row2 r0, r1;
};

__device__ float mat22_trace(Matrix22 m) {
    return m.r0.a + m.r1.b;
}

__global__ void matrix_trace(float *out, Matrix22 *mats, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = mat22_trace(mats[tid]);
    }
}

// ------------------------------------------------------------------
// Struct field update in loop: accumulate into struct fields.
// Loop-carried struct field values must survive writeback.

struct Stats {
    int count;
    int sum;
};

__global__ void struct_accum_loop(int *out_count, int *out_sum,
                                   int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s;
        s.count = 0;
        s.sum   = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] > 0) {
                s.count++;
                s.sum += data[i];
            }
        }
        out_count[0] = s.count;
        out_sum[0]   = s.sum;
    }
}

// ------------------------------------------------------------------
// Array of structs: each thread writes a struct field.
// Then a second pass reads back a different field.

struct Pair2 {
    int lo, hi;
};

__global__ void aos_write_read(int *out, Pair2 *pairs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = pairs[tid].hi - pairs[tid].lo;
    }
}

// ------------------------------------------------------------------
// Struct returned by device function (inlined).
// The returned struct must have all fields accessible.

struct MinMax {
    float mn, mx;
};

__device__ MinMax compute_minmax(float a, float b) {
    MinMax r;
    r.mn = (a < b) ? a : b;
    r.mx = (a > b) ? a : b;
    return r;
}

__global__ void struct_return_inline(float *out_mn, float *out_mx,
                                      float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        MinMax mm = compute_minmax(a[tid], b[tid]);
        out_mn[tid] = mm.mn;
        out_mx[tid] = mm.mx;
    }
}

// ------------------------------------------------------------------
// Struct with int and pointer field passed to device fn.
// Tests pointer field binding across inline boundary.

struct Slice {
    int   len;
    int  *ptr;
};

__device__ int slice_sum(Slice s) {
    int total = 0;
    for (int i = 0; i < s.len; i++) {
        total += s.ptr[i];
    }
    return total;
}

__global__ void struct_ptr_field_inline(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Slice s;
        s.len = n;
        s.ptr = data;
        out[0] = slice_sum(s);
    }
}
