// Probe: tricky parameter mutation patterns,
// parameter used as both limit and accumulator,
// same register reused after param is no longer needed

__global__ void param_reuse(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // n used as bound in loop, then as output offset
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += i;
        }
        out[tid] = sum + n;  // n used again after loop
    }
}

// Parameter modified locally
__global__ void param_shadow(float *out, float alpha, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Compute something that shadows alpha
        if (alpha < 0.0f) alpha = -alpha;  // abs(alpha)
        if (alpha > 1.0f) alpha = 1.0f;   // clamp
        out[tid] = alpha * (float)tid;
    }
}

// Multiple params, some modified
__global__ void multi_param_modify(int *out, int a, int b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        a = a * 2 + 1;
        b = b - a;
        out[tid] = a + b * tid;
    }
}

// Param used as array index
__global__ void param_as_index(float *out, float *table, int base, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = (base + tid) & 31;  // base modified by tid
        out[tid] = table[idx];
    }
}
