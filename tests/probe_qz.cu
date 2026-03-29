// Probe: CUDA built-in vector types (float3, float4, int2, int4),
// struct initializer list syntax, and vector type arithmetic.

// ------------------------------------------------------------------
// float3 arithmetic: element-wise operations.

__global__ void float3_ops(float3 *out, float3 *a, float3 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float3 va = a[tid];
        float3 vb = b[tid];
        float3 r;
        r.x = va.x + vb.x;
        r.y = va.y + vb.y;
        r.z = va.z + vb.z;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// float4 dot product.

__device__ float dot4f(float4 a, float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__global__ void float4_dots(float *out, float4 *a, float4 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dot4f(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// int2 pair operations.

__global__ void int2_ops(int2 *out, int2 *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int2 v = data[tid];
        int2 r;
        r.x = v.x + v.y;
        r.y = v.x - v.y;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// int4 min/max on each component.

__global__ void int4_minmax(int4 *out_min, int4 *out_max,
                             int4 *a, int4 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int4 va = a[tid];
        int4 vb = b[tid];
        int4 mn, mx;
        mn.x = va.x < vb.x ? va.x : vb.x;
        mn.y = va.y < vb.y ? va.y : vb.y;
        mn.z = va.z < vb.z ? va.z : vb.z;
        mn.w = va.w < vb.w ? va.w : vb.w;
        mx.x = va.x > vb.x ? va.x : vb.x;
        mx.y = va.y > vb.y ? va.y : vb.y;
        mx.z = va.z > vb.z ? va.z : vb.z;
        mx.w = va.w > vb.w ? va.w : vb.w;
        out_min[tid] = mn;
        out_max[tid] = mx;
    }
}

// ------------------------------------------------------------------
// float2 complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

__global__ void complex_mul(float2 *out, float2 *a, float2 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 va = a[tid];
        float2 vb = b[tid];
        float2 r;
        r.x = va.x * vb.x - va.y * vb.y;
        r.y = va.x * vb.y + va.y * vb.x;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// make_float3 / make_int2 constructors.

__global__ void make_vec_kernel(float3 *out_f3, int2 *out_i2,
                                 float *fin, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float3 f3 = make_float3(fin[tid], fin[tid]*2.0f, fin[tid]*3.0f);
        int2   i2 = make_int2(iin[tid], iin[tid]+1);
        out_f3[tid] = f3;
        out_i2[tid] = i2;
    }
}

// ------------------------------------------------------------------
// uint2 bitwise ops.

__global__ void uint2_bitops(uint2 *out, uint2 *a, uint2 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint2 va = a[tid];
        uint2 vb = b[tid];
        uint2 r;
        r.x = va.x ^ vb.x;
        r.y = va.y & vb.y;
        out[tid] = r;
    }
}
