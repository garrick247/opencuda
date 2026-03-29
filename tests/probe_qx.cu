// Probe: short unrollable loops with struct fields, optimizer CSE on
// repeated global struct field addresses, and various __device__ function
// return patterns.

// ------------------------------------------------------------------
// Short loop (trip count <= 16): should be unrolled.

__global__ void dot4(float *out, float *A, float *B, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 4;
        float s = 0.0f;
        for (int i = 0; i < 4; i++) {
            s += A[base + i] * B[base + i];
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Short loop (trip count = 8) with struct field update.

struct V4 {
    float x, y, z, w;
};

__device__ float v4_dot(V4 a, V4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void v4_dots(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 8;
        V4 a, b;
        a.x = data[base + 0]; a.y = data[base + 1];
        a.z = data[base + 2]; a.w = data[base + 3];
        b.x = data[base + 4]; b.y = data[base + 5];
        b.z = data[base + 6]; b.w = data[base + 7];
        out[tid] = v4_dot(a, b);
    }
}

// ------------------------------------------------------------------
// Optimizer: CSE on repeated access to global struct field.
// g_config.scale is accessed multiple times → should CSE to one load.

struct Config {
    float scale;
    float bias;
    int   mode;
};

__device__ Config g_config;

__global__ void apply_config(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Multiple reads of same field — CSE should deduplicate
        float s = g_config.scale;
        float b = g_config.bias;
        int   m = g_config.mode;
        float r;
        if (m == 0) {
            r = v * s + b;
        } else if (m == 1) {
            r = v * s * s + b;
        } else {
            r = v + b * s;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Return from different branches (not just early return).

__device__ float classify_val(float v, float lo, float hi, float mid) {
    if (v < lo) {
        return -1.0f;
    } else if (v > hi) {
        return 1.0f;
    } else if (v < mid) {
        return -0.5f;
    } else {
        return 0.5f;
    }
}

__global__ void classify_kernel(float *out, float *data, float lo, float hi,
                                 float mid, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify_val(data[tid], lo, hi, mid);
    }
}

// ------------------------------------------------------------------
// Loop unrolling: trip count = 16, uses struct field.

struct Acc16 {
    float sum;
    int   cnt;
};

__global__ void acc16_kernel(float *out_sum, int *out_cnt, float *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc16 a;
        a.sum = 0.0f;
        a.cnt = 0;
        for (int i = 0; i < 16; i++) {
            a.sum += in[i];
            a.cnt++;
        }
        out_sum[0] = a.sum;
        out_cnt[0] = a.cnt;
    }
}

// ------------------------------------------------------------------
// Optimizer correctness: constant folding through struct field.

__global__ void const_fold_struct(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        V4 v;
        v.x = 1.0f; v.y = 2.0f; v.z = 3.0f; v.w = 4.0f;
        // These should constant-fold: 1+2+3+4=10
        float s = v.x + v.y + v.z + v.w;
        out[tid] = s * (float)tid;
    }
}
