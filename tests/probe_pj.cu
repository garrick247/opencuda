// Probe: struct-by-value inlining patterns + assignment-in-condition variants.
// Exercises the fixes from v0.64 in more complex configurations.

// ------------------------------------------------------------------
// Struct passed by value from a LOCAL variable (not array element).
// Tests the existing-field-vars binding path.

struct Vec2 {
    float x, y;
};

__device__ float dot2(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

__global__ void local_struct_pass(float *out, float *ax, float *ay,
                                   float *bx, float *by, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 a, b;
        a.x = ax[tid]; a.y = ay[tid];
        b.x = bx[tid]; b.y = by[tid];
        out[tid] = dot2(a, b);
    }
}

// ------------------------------------------------------------------
// Struct passed from array element.  Tests the pointer-load path added in v0.64.

__global__ void array_struct_pass(float *out, Vec2 *vecs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // vecs[tid] is a struct array element — passed by value
        float len2 = dot2(vecs[tid], vecs[tid]);
        out[tid] = len2;
    }
}

// ------------------------------------------------------------------
// Assignment-in-condition: integer assignment in && with further use.
// `while (i < n && (sum += arr[i]) < limit)` — compound assign in &&.

__global__ void compound_assign_in_and(int *out, int *arr, int n, int limit) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        // sum += arr[i] rebinds sum inside land_rhs — must be multi-def after fix
        while (i < n && (sum += arr[i]) < limit) {
            i++;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Assignment-in-|| condition: `while ((v = a[i]) != 0 || (w = b[i]) != 0)`.
// Tests the lor_skip copy-for-rebound path in _parse_or_expr.

__global__ void assign_in_or(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0, v = 0, w = 0;
        int i = 0;
        while (i < n && ((v = a[i]) != 0 || (w = b[i]) != 0)) {
            sum += v + w;
            i++;
            v = 0; w = 0;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Float assignment-in-condition.
// `while (i < n && (f = data[i]) > 0.0f)` — float rebound in land_rhs.

__global__ void float_assign_in_and(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f, f = 0.0f;
        int i = 0;
        while (i < n && (f = data[i]) > 0.0f) {
            sum += f;
            i++;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Nested struct passed by value from an array element.
// `Outer` contains an `Inner` field — exercises recursive field loading.

struct Inner {
    float u, v;
};

struct Outer {
    int id;
    Inner uv;
};

__device__ float use_inner(Inner q) {
    return q.u + q.v;
}

__device__ float use_outer(Outer o) {
    return (float)o.id + o.uv.u + o.uv.v;
}

__global__ void nested_struct_pass(float *out, Outer *objs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Both use_inner(objs[tid].uv) and use_outer(objs[tid]) exercise
        // nested struct field loading
        float a = use_inner(objs[tid].uv);
        float b = use_outer(objs[tid]);
        out[tid] = a + b;
    }
}
