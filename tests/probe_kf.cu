// Probe: float accumulation with int widening (float += int),
// pre-decrement in while condition (while (--count > 0)),
// device function param names collide with caller kernel params,
// computed loop bounds, fused multiply-add pattern

// float += int: int must be widened before add
__global__ void float_plus_int(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += in[i];    // int widened to float before add
        }
        *out = acc;
    }
}

// Pre-decrement in while condition
__global__ void countdown(int *out, int start) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = start;
        int sum = 0;
        while (--count > 0) {
            sum += count;    // sum = (start-1) + (start-2) + ... + 1
        }
        *out = sum;
    }
}

// Device function with param names colliding with kernel params
// 'n' appears in both kernel and device fn
__device__ int repeat_add(int val, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += val;
    }
    return total;
}

__global__ void param_name_collision(int *out, int *in, int n, int repeats) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Device fn 'n' param gets 'repeats', kernel 'n' is array size
        out[tid] = repeat_add(in[tid], repeats);
    }
}

// FMA pattern: a*b + c without explicit FMA intrinsic
// Tests that the mul then add sequence is correct
__global__ void manual_fma(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float result = a[tid] * b[tid] + c[tid];
        out[tid] = result;
    }
}

// Computed loop bound: limit = n - 2 (excludes last two)
__global__ void computed_bound(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int limit = n - 2;
        int sum = 0;
        for (int i = 1; i < limit; i++) {   // skip first and last two
            sum += in[i];
        }
        *out = sum;
    }
}
