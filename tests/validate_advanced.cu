// Advanced patterns for runtime validation: device functions, control flow,
// type conversions, and mixed-precision computation.

// Device function inlined into kernel
__device__ float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

__global__ void lerp_kernel(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = lerp(a[gid], b[gid], 0.3f);
}

// Ternary chain
__global__ void classify(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = a[gid];
        out[gid] = (v < -10) ? -2 :
                   (v <   0) ? -1 :
                   (v ==  0) ?  0 :
                   (v <  10) ?  1 : 2;
    }
}

// Loop accumulation with conditional
__global__ void cond_accum(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int s = 0;
        for (int i = 0; i < 16; i++) {
            int v = a[(gid + i) % n];
            if (v > 0) s += v;
        }
        out[gid] = s;
    }
}

// Float to int conversion
__global__ void float_to_int(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float f = (float)a[gid] * 0.1f + 0.5f;
        out[gid] = (int)f;
    }
}

// Integer division and modulo
__global__ void divmod(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = a[gid];
        int d = (b[gid] != 0) ? b[gid] : 1;
        out[gid] = (v / d) * 1000 + (v % d);
    }
}

// Warp prefix sum (inclusive scan via shfl_up)
__global__ void warp_scan(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = (gid < n) ? a[gid] : 0;
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if (lane >= d) v += t;
    }
    if (gid < n) out[gid] = v;
}

// Multi-output: compute both sum and product into interleaved output
__global__ void sum_and_diff(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // Store sum in even positions, diff in odd positions
        // But we only have one output array, so encode as sum*1000 + abs(diff)
        int s = a[gid] + b[gid];
        int d = a[gid] - b[gid];
        if (d < 0) d = -d;
        out[gid] = s * 1000 + d;
    }
}
