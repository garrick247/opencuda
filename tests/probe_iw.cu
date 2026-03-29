// Probe: early-exit patterns — return from inside loops and conditionals,
// multiple return points in device functions,
// return with non-trivial expression,
// return after mutations (value at return point matters)

// Early exit: return first element >= threshold
__device__ int first_ge_thresh(int *arr, int n, int thresh) {
    for (int i = 0; i < n; i++) {
        if (arr[i] >= thresh) return i;
    }
    return -1;
}

__global__ void use_first_ge(int *out, int *arr, int n, int thresh) {
    int tid = threadIdx.x;
    if (tid == 0) {
        *out = first_ge_thresh(arr, n, thresh);
    }
}

// Multiple return points in conditional chain
__device__ float classify(float v) {
    if (v < -1.0f) return -2.0f;
    if (v < 0.0f)  return -1.0f;
    if (v < 1.0f)  return 0.0f;
    if (v < 2.0f)  return 1.0f;
    return 2.0f;
}

__global__ void use_classify(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify(in[tid]);
    }
}

// Return with expression computed from arguments
__device__ int dot2(int ax, int ay, int bx, int by) {
    return ax * bx + ay * by;
}

__global__ void use_dot2(int *out, int *ax, int *ay, int *bx, int *by, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dot2(ax[tid], ay[tid], bx[tid], by[tid]);
    }
}

// Return inside nested if: value set before return
__device__ int bounded_div(int a, int b) {
    if (b == 0) return 0;
    int q = a / b;
    if (q > 1000) return 1000;
    if (q < -1000) return -1000;
    return q;
}

__global__ void use_bounded_div(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = bounded_div(a[tid], b[tid]);
    }
}
