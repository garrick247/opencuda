// Probe: inline device function edge cases — mutual recursion-style
// chains, cross-parameter aliasing, large struct return, and
// device fn that uses __syncthreads (should be inlined in-place).

// ------------------------------------------------------------------
// Device fn that modifies two output pointers.

__device__ void swap_float(float *a, float *b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void two_ptr_out(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid + 1 < n) {
        float x = in[tid];
        float y = in[tid + 1];
        swap_float(&x, &y);
        out[tid]     = x;
        out[tid + 1] = y;
    }
}

// ------------------------------------------------------------------
// Chain: A calls B calls C.

__device__ int triply_nested(int x) {
    return x * x + 1;
}

__device__ int double_nested(int x) {
    return triply_nested(x) + x;
}

__device__ int single_nested(int x) {
    return double_nested(x) * 2;
}

__global__ void call_chain_depth3(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = single_nested(in[tid]);
    }
}

// ------------------------------------------------------------------
// Device fn called with local struct output (write-through pointer).

struct Stats {
    float mean;
    float variance;
    int count;
};

__device__ void compute_stats(const float *data, int len, Stats *s) {
    float sum = 0.0f, sum2 = 0.0f;
    int cnt = 0;
    for (int i = 0; i < len; i++) {
        float v = data[i];
        if (v >= 0.0f) {
            sum += v;
            sum2 += v * v;
            cnt++;
        }
    }
    s->count = cnt;
    s->mean = (cnt > 0) ? sum / (float)cnt : 0.0f;
    float avg2 = (cnt > 0) ? sum2 / (float)cnt : 0.0f;
    s->variance = avg2 - s->mean * s->mean;
}

__global__ void stats_kernel(float *out_mean, float *out_var,
                               float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Stats s;
        compute_stats(&in[tid * stride], stride, &s);
        out_mean[tid] = s.mean;
        out_var[tid]  = s.variance;
    }
}

// ------------------------------------------------------------------
// Device fn that takes a const reference-like pattern (const T *).

__device__ float dot4(const float *a, const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

__global__ void batch_dot4(float *out, float *mat, float *vec, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dot4(&mat[tid * 4], vec);
    }
}

// ------------------------------------------------------------------
// Device fn that selects between two code paths based on runtime flag.

__device__ float select_op(float a, float b, int op) {
    if (op == 0) return a + b;
    if (op == 1) return a - b;
    if (op == 2) return a * b;
    if (op == 3) return (b != 0.0f) ? (a / b) : 0.0f;
    return 0.0f;
}

__global__ void dispatch_op(float *out, float *a, float *b, int *ops, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = select_op(a[tid], b[tid], ops[tid]);
    }
}
