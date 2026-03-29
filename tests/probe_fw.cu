// Probe: multiple typedef chains, forward-declared struct, typedef-ed function pointer
// Also: typedef struct { ... } Alias; pattern (anonymous struct with typedef)

typedef struct {
    float x, y;
} float2_t;

typedef struct {
    float x, y, z;
} float3_t;

typedef struct {
    float x, y, z, w;
} float4_t;

__device__ float2_t make_float2_t(float x, float y) {
    float2_t r;
    r.x = x; r.y = y;
    return r;
}

__device__ float dot2(float2_t a, float2_t b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float3_t cross3(float3_t a, float3_t b) {
    float3_t r;
    r.x = a.y * b.z - a.z * b.y;
    r.y = a.z * b.x - a.x * b.z;
    r.z = a.x * b.y - a.y * b.x;
    return r;
}

__global__ void vec_ops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 3;
        float3_t a;
        a.x = in[base]; a.y = in[base+1]; a.z = in[base+2];
        float3_t b;
        b.x = in[base+2]; b.y = in[base+1]; b.z = in[base];
        float3_t c = cross3(a, b);
        out[tid] = c.x + c.y + c.z;
    }
}

__global__ void dot_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 2;
        float2_t a = make_float2_t(in[base], in[base+1]);
        float2_t b = make_float2_t(in[base+1], in[base]);
        out[tid] = dot2(a, b);
    }
}
