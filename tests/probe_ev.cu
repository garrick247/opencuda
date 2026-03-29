// Probe: multiple nested if-else-if chains, early returns in __device__ functions
// Also: deeply nested conditionals, if with compound && / || in condition

__device__ int classify(float v) {
    if (v < -1.0f) return -2;
    else if (v < 0.0f) return -1;
    else if (v == 0.0f) return 0;
    else if (v <= 1.0f) return 1;
    else return 2;
}

__device__ float safe_sqrt(float v) {
    if (v < 0.0f) return 0.0f;
    return v * v;  // approximate: v^2 as placeholder
}

__global__ void nested_cond(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int cls = classify(v);
        float r;
        if (cls < 0) {
            if (cls == -2) {
                r = -2.0f;
            } else {
                r = -1.0f;
            }
        } else if (cls == 0) {
            r = 0.0f;
        } else {
            if (cls == 1) {
                r = safe_sqrt(v);
            } else {
                r = 1.0f;
            }
        }
        out[tid] = r;
    }
}

// Deep AND/OR conditions
__global__ void compound_cond(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid], z = c[tid];
        int result = 0;
        if (x > 0 && y > 0 && z > 0) {
            result = 1;
        } else if (x < 0 || y < 0 || z < 0) {
            result = -1;
        } else if ((x == 0 && y == 0) || z == 0) {
            result = 2;
        }
        out[tid] = result;
    }
}
