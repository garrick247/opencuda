// Probe: nested device calls (device calling device),
// float-to-int truncation in accumulation,
// struct pointer field access,
// loop with device function producing intermediate value used twice

// ---- nested device calls ----

__device__ int clamp_val(int v, int lo, int hi) {
    int result = v;
    if (result < lo) result = lo;
    if (result > hi) result = hi;
    return result;
}

__device__ int normalize(int v, int scale) {
    int clamped = clamp_val(v, 0, scale);
    return clamped * 100 / scale;  // returns 0-100
}

__global__ void nested_device_call(int *out, int *in, int scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = normalize(in[tid], scale);
    }
}

// ---- float-to-int truncation in loop ----
__global__ void float_to_int_accum(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            int trunc = (int)in[i];   // truncate toward zero
            total += trunc;
        }
        *out = total;
    }
}

// ---- struct with pointer passed to kernel ----
struct Pair {
    int x;
    int y;
};

__global__ void struct_field_sum(int *out, struct Pair *pairs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = pairs[tid].x + pairs[tid].y;
    }
}

// ---- device function result used twice in same expression ----
__device__ int abs_val(int x) {
    int r = x;
    if (r < 0) r = -r;
    return r;
}

__global__ void double_use_device(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = abs_val(v);
        int b = abs_val(v - 10);
        out[tid] = a + b;
    }
}
