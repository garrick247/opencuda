// Probe: local nested struct variable field access, __device__ global scalars,
// __device__ function early return paths, and misc arithmetic edge cases.

// ------------------------------------------------------------------
// Global __device__ scalar (non-array, non-struct) write from kernel.

__device__ int g_flag;
__device__ float g_accum;

__global__ void set_flag(int v) {
    if (threadIdx.x == 0) {
        g_flag = v;
        g_accum = (float)v * 3.14159f;
    }
}

__global__ void read_flag(int *out_flag, float *out_accum) {
    if (threadIdx.x == 0) {
        out_flag[0] = g_flag;
        out_accum[0] = g_accum;
    }
}

// ------------------------------------------------------------------
// Local nested struct variable (by value) field access and assignment.

struct Inner {
    float u, v;
};

struct Outer {
    Inner a;
    Inner b;
    float w;
};

__global__ void local_nested_struct(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Outer s;
        s.a.u = in[tid * 4 + 0];
        s.a.v = in[tid * 4 + 1];
        s.b.u = in[tid * 4 + 2];
        s.b.v = in[tid * 4 + 3];
        s.w = s.a.u + s.a.v + s.b.u + s.b.v;
        out[tid] = s.w;
    }
}

// ------------------------------------------------------------------
// __device__ function: early return inside nested if.

__device__ float safe_divide(float a, float b) {
    if (b == 0.0f) {
        return 0.0f;
    }
    if (a == 0.0f) {
        return 0.0f;
    }
    return a / b;
}

__global__ void safe_div_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_divide(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Ternary in complex expression.

__global__ void ternary_complex(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = data[tid];
        int y = (x > 0) ? (x * 2 + 1) : ((x < 0) ? (x * 3 - 1) : 0);
        out[tid] = y;
    }
}

// ------------------------------------------------------------------
// Prefix/postfix ++ on struct fields.

struct Counter {
    int hits;
    int misses;
};

__global__ void count_hits(int *out_h, int *out_m, int *data, int threshold, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Counter c;
        c.hits = 0;
        c.misses = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] >= threshold) {
                c.hits++;
            } else {
                c.misses++;
            }
        }
        out_h[0] = c.hits;
        out_m[0] = c.misses;
    }
}

// ------------------------------------------------------------------
// Compound assignment on struct field in a loop.

__global__ void accumulate_fields(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Inner acc;
        acc.u = 0.0f;
        acc.v = 0.0f;
        for (int i = 0; i < n; i++) {
            acc.u += in[i * 2 + 0];
            acc.v += in[i * 2 + 1];
        }
        out[0] = acc.u;
        out[1] = acc.v;
    }
}

// ------------------------------------------------------------------
// __device__ scalar: conditional update.

__device__ int g_max_val;

__global__ void update_max(int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int local_max = data[0];
        for (int i = 1; i < n; i++) {
            if (data[i] > local_max) {
                local_max = data[i];
            }
        }
        g_max_val = local_max;
    }
}

__global__ void read_max(int *out) {
    if (threadIdx.x == 0) {
        out[0] = g_max_val;
    }
}
