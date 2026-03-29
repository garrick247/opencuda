// Probe: float/int type promotion in ternary, #pragma unroll,
// struct with array member returned from device fn, and
// complex inline device function interactions.

// ------------------------------------------------------------------
// #pragma unroll — should be parsed (as comment/ignored) and loop still works.

__global__ void pragma_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc += v + i;
        }
        out[tid] = acc;  // 4v + 6
    }
}

// ------------------------------------------------------------------
// #pragma unroll with explicit count.

__global__ void pragma_unroll_n(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float acc = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            acc += v * (float)i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Ternary with int and float arms — implicit promotion to float.

__global__ void ternary_type_promote(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // int arm and float arm — result should be float
        float r = (v > 0) ? v : -1.0f;   // int arm promotes to float
        float s = (v > 0) ? 1.0f : v;    // int arm in false branch
        out[tid] = r + s;
    }
}

// ------------------------------------------------------------------
// Struct with array member returned from device fn.

struct Array4 {
    int data[4];
};

__device__ struct Array4 make_array4(int base) {
    struct Array4 r;
    r.data[0] = base;
    r.data[1] = base + 1;
    r.data[2] = base + 2;
    r.data[3] = base + 3;
    return r;
}

__global__ void struct_array_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Array4 a = make_array4(in[tid]);
        out[tid] = a.data[0] + a.data[1] + a.data[2] + a.data[3];
        // = base + (base+1) + (base+2) + (base+3) = 4*base + 6
    }
}

// ------------------------------------------------------------------
// Nested device fn calls (A→B→C) with different return types.

__device__ int level_c(int v) { return v * v; }
__device__ float level_b(int v) { return sqrtf((float)level_c(v)); }
__device__ float level_a(int v, float bias) { return level_b(v) + bias; }

__global__ void three_level_call(float *out, int *in, float *bias, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = level_a(in[tid], bias[tid]);
    }
}

// ------------------------------------------------------------------
// Complex expression with function call in condition.

__device__ int check_range(int v, int lo, int hi) {
    return (v >= lo && v <= hi) ? 1 : 0;
}

__global__ void call_in_condition(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        // Function call as condition
        if (check_range(v, 0, 100)) r += 1;
        if (check_range(v, 50, 150)) r += 2;
        if (check_range(v, -10, 10)) r += 4;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Attribute syntax — should be parseable (ignored).

__global__ __attribute__((reqd_work_group_size(256, 1, 1)))
void attr_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] + 1;
    }
}
