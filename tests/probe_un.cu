// Probe: device functions with many args, struct with array members,
// nested struct access, and complex global access patterns.

struct Filter {
    float coeffs[8];
    int len;
    float scale;
};

struct Node {
    int parent;
    int left;
    int right;
    float value;
};

// ------------------------------------------------------------------
// Device function with 8+ arguments.

__device__ float poly8(float x,
                        float c0, float c1, float c2, float c3,
                        float c4, float c5, float c6, float c7) {
    return c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*(c6 + x*c7))))));
}

__global__ void eval_poly8(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = poly8(x, 1.0f, 2.0f, -1.0f, 0.5f,
                            0.1f, -0.05f, 0.01f, 0.001f);
    }
}

// ------------------------------------------------------------------
// Device function with struct-by-value + array member.

__device__ float apply_filter(Filter f, float *data, int offset) {
    float acc = 0.0f;
    for (int i = 0; i < f.len; i++) {
        int src = offset - i;
        if (src >= 0) {
            acc += f.coeffs[i] * data[src];
        }
    }
    return acc * f.scale;
}

__global__ void filter_kernel(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        Filter f;
        f.coeffs[0] = 0.25f; f.coeffs[1] = 0.25f;
        f.coeffs[2] = 0.25f; f.coeffs[3] = 0.25f;
        f.coeffs[4] = 0.0f;  f.coeffs[5] = 0.0f;
        f.coeffs[6] = 0.0f;  f.coeffs[7] = 0.0f;
        f.len = 4;
        f.scale = 1.0f;
        out[tid] = apply_filter(f, in, tid);
    }
}

// ------------------------------------------------------------------
// Tree traversal via Node struct array.

__global__ void tree_sum(float *out, Node *nodes, int root, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Follow left spine from root and accumulate values
        float acc = 0.0f;
        int cur = root;
        for (int depth = 0; depth < 8 && cur >= 0; depth++) {
            acc += nodes[cur].value;
            cur = nodes[cur].left;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Global __device__ struct array — read and write.

struct Point2D {
    float x;
    float y;
};

__device__ Point2D g_points[64];

__global__ void scatter_points(float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 64) {
        g_points[tid].x = xs[tid];
        g_points[tid].y = ys[tid];
    }
}

__global__ void gather_points(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 64) {
        out[tid * 2]     = g_points[tid].x;
        out[tid * 2 + 1] = g_points[tid].y;
    }
}
