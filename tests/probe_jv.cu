// Probe: float4/int2/float2 vector types with component access,
// vector arithmetic, struct with mixed types,
// device function taking/returning vector type

// float4 component-wise operations
__global__ void vec4_ops(float4 *out, float4 *a, float4 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float4 av = a[tid];
        float4 bv = b[tid];
        float4 result;
        result.x = av.x + bv.x;
        result.y = av.y - bv.y;
        result.z = av.z * bv.z;
        result.w = av.w / bv.w;
        out[tid] = result;
    }
}

// int2 as pair of integers
__global__ void int2_swap(int2 *out, int2 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int2 v = in[tid];
        int2 swapped;
        swapped.x = v.y;
        swapped.y = v.x;
        out[tid] = swapped;
    }
}

// float2 magnitude squared
__global__ void float2_magsq(float *out, float2 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 v = in[tid];
        out[tid] = v.x * v.x + v.y * v.y;
    }
}

// Device function accepting float2 and returning float2
__device__ float2 scale2(float2 v, float s) {
    float2 result;
    result.x = v.x * s;
    result.y = v.y * s;
    return result;
}

__global__ void vec2_scale(float2 *out, float2 *in, float s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 v = in[tid];
        float2 r = scale2(v, s);
        out[tid] = r;
    }
}
