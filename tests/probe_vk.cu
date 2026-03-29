// Probe: CUDA vector types — float2, float4, int2, int4.
// These are structs with named fields (x, y, z, w).

// ------------------------------------------------------------------
// float2: basic load, store, and field access.

__global__ void float2_ops(float2 *out, float2 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 v = in[tid];
        float2 r;
        r.x = v.x + v.y;
        r.y = v.x - v.y;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// float4: RGBA-like access.

__global__ void float4_ops(float4 *out, float4 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float4 v = in[tid];
        float4 r;
        r.x = v.x * 2.0f;
        r.y = v.y * 2.0f;
        r.z = v.z * 2.0f;
        r.w = v.w * 2.0f;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// int2: coordinate arithmetic.

__global__ void int2_ops(int2 *out, int2 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int2 a = in[tid * 2];
        int2 b = in[tid * 2 + 1];
        int2 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// float4 dot product.

__device__ float dot4f(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void float4_dot(float *out, float4 *a, float4 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dot4f(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// int4: bitfield-like packing.

__global__ void int4_pack(int *out, int4 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int4 v = in[tid];
        // Pack 4 bytes into one int (assuming values fit in 8 bits)
        int packed = ((v.x & 0xFF))
                   | ((v.y & 0xFF) << 8)
                   | ((v.z & 0xFF) << 16)
                   | ((v.w & 0xFF) << 24);
        out[tid] = packed;
    }
}

// ------------------------------------------------------------------
// Make a float2 from scalar values and store.

__global__ void make_float2_kernel(float2 *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 v;
        v.x = xs[tid];
        v.y = ys[tid];
        out[tid] = v;
    }
}
