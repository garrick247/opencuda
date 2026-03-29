// Probe: typedef struct names without struct keyword, const pointer
// parameters, restrict pointer parameters, and forward-declared types.

typedef struct { float x, y; } float2_t;
typedef struct { float x, y, z; } float3_t;
typedef struct { int lo, hi; } range_t;

// ------------------------------------------------------------------
// Typedef struct used as parameter type (no 'struct' keyword).

__global__ void float2_sum(float *out, float2_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2_t v = in[tid];
        out[tid] = v.x + v.y;
    }
}

// ------------------------------------------------------------------
// Device function taking typedef struct by value.

__device__ float float3_dot(float3_t a, float3_t b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__global__ void float3_dot_kernel(float *out, float3_t *a, float3_t *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = float3_dot(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// const pointer parameter.

__device__ float const_ptr_sum(const float *data, int count) {
    float acc = 0.0f;
    for (int i = 0; i < count; i++) {
        acc += data[i];
    }
    return acc;
}

__global__ void const_ptr_kernel(float *out, const float *in, int n, int stride) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = const_ptr_sum(in + tid * stride, stride);
    }
}

// ------------------------------------------------------------------
// range_t struct for bounds checking.

__device__ int in_range(range_t r, int v) {
    return (v >= r.lo && v <= r.hi) ? 1 : 0;
}

__global__ void range_check(int *out, range_t *ranges, int *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in_range(ranges[tid], vals[tid]);
    }
}

// ------------------------------------------------------------------
// typedef struct pointer parameter.

__device__ void float2_scale(float2_t *v, float s) {
    v->x *= s;
    v->y *= s;
}

__global__ void scale_float2(float2_t *vecs, float s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2_scale(&vecs[tid], s);
    }
}

// ------------------------------------------------------------------
// Typedef struct array returned via output pointer.

__device__ void make_range(int center, int half, range_t *out) {
    out->lo = center - half;
    out->hi = center + half;
}

__global__ void build_ranges(range_t *out, int *centers, int half, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        make_range(centers[tid], half, &out[tid]);
    }
}

// ------------------------------------------------------------------
// Multiple typedef structs in one kernel.

__global__ void multi_typedef(float *out, float2_t *a2, float3_t *a3, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2_t v2 = a2[tid];
        float3_t v3 = a3[tid];
        out[tid] = v2.x + v2.y + v3.x + v3.y + v3.z;
    }
}
