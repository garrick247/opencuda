// Probe: Deeply nested control flow + early returns in __device__ functions
// - Device function with return in middle of if-else chain
// - Device function where all branches return
// - Device function called from within switch case
// - Nested function calls where outer function has early return
// - Function with return in loop body

__device__ int classify(int x) {
    if (x < 0) return -1;
    if (x == 0) return 0;
    if (x < 10) return 1;
    if (x < 100) return 2;
    return 3;
}

__global__ void classify_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify(in[tid]);
    }
}

__device__ float safe_div(float a, float b) {
    if (b == 0.0f) return 0.0f;
    return a / b;
}

__global__ void safe_div_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_div(a[tid], b[tid]);
    }
}

// Device function called from within a switch
__device__ int score(int category, int value) {
    if (category == 0) return value;
    if (category == 1) return value * 2;
    return value / 2;
}

__global__ void switch_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        switch (v % 4) {
            case 0: result = score(0, v); break;
            case 1: result = score(1, v); break;
            case 2: result = score(2, v); break;
            default: result = 0; break;
        }
        out[tid] = result;
    }
}

// Nested calls where outer has early return
__device__ float clamp(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

__device__ float normalize(float x, float mn, float mx) {
    float range = mx - mn;
    if (range == 0.0f) return 0.5f;
    return clamp((x - mn) / range, 0.0f, 1.0f);
}

__global__ void normalize_kernel(float *out, float *in, float mn, float mx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = normalize(in[tid], mn, mx);
    }
}

// Return in loop
__device__ int first_positive(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] > 0) return arr[i];
    }
    return -1;
}

__global__ void first_pos_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = first_positive(in, n);
    }
}
