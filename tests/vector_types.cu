// Regression: CUDA built-in vector types (float2, float3, float4, int2, etc.)
// Each field (x/y/z/w) is accessed via dot notation and maps to a separate
// scalar PTX register. No packed struct layout needed.
// Without fix: ParseError "undefined variable 'float3'"

__global__ void float3_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float3 v;
        v.x = in[tid * 3 + 0];
        v.y = in[tid * 3 + 1];
        v.z = in[tid * 3 + 2];
        // Simple per-component scale
        v.x = v.x * 2.0f;
        v.y = v.y * 3.0f;
        v.z = v.z + 1.0f;
        out[tid * 3 + 0] = v.x;
        out[tid * 3 + 1] = v.y;
        out[tid * 3 + 2] = v.z;
    }
}

__global__ void float2_dot(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 u;
        float2 w;
        u.x = a[tid * 2 + 0];
        u.y = a[tid * 2 + 1];
        w.x = b[tid * 2 + 0];
        w.y = b[tid * 2 + 1];
        float dot = u.x * w.x + u.y * w.y;
        out[tid] = dot;
    }
}

__global__ void int2_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int2 p;
        p.x = in[tid * 2 + 0];
        p.y = in[tid * 2 + 1];
        int sum = p.x + p.y;
        out[tid] = sum;
    }
}
