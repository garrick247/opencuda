// Probe: real-world GPU computing patterns that push remaining limits —
// recursive tree traversal, BVH intersection, radix sort building blocks,
// warp-cooperative scan with bank conflict avoidance, double-buffered
// shared memory, grid-stride loop with atomics, and complex expressions
// with many temporaries.

// ------------------------------------------------------------------
// Recursive tree traversal (uses new recursive device func support).

struct BVHNode {
    float min_x, max_x;
    int left;   // -1 = leaf
    int right;
};

__device__ int bvh_intersect(struct BVHNode *nodes, float query, int node_idx) {
    if (node_idx < 0) return 0;
    struct BVHNode n = nodes[node_idx];
    if (query < n.min_x || query > n.max_x) return 0;
    if (n.left < 0) return 1;  // leaf hit
    return bvh_intersect(nodes, query, n.left)
         + bvh_intersect(nodes, query, n.right);
}

__global__ void bvh_kernel(int *out, struct BVHNode *nodes, float *queries, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = bvh_intersect(nodes, queries[tid], 0);
}

// ------------------------------------------------------------------
// Radix sort building block: count bits at position.

__global__ void radix_count(int *counts, int *keys, int bit, int n) {
    __shared__ int local_count[2];
    int tid = threadIdx.x;
    if (tid < 2) local_count[tid] = 0;
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        int b = (keys[gid] >> bit) & 1;
        atomicAdd(&local_count[b], 1);
    }
    __syncthreads();
    if (tid < 2) atomicAdd(&counts[blockIdx.x * 2 + tid], local_count[tid]);
}

// ------------------------------------------------------------------
// Double-buffered shared memory ping-pong.

__global__ void double_buffer(float *out, float *in, int n, int iters) {
    __shared__ float buf0[256];
    __shared__ float buf1[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load into buf0
    buf0[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    for (int it = 0; it < iters; it++) {
        if (it % 2 == 0) {
            // Read buf0, write buf1
            float left  = (tid > 0) ? buf0[tid-1] : 0.0f;
            float right = (tid < 255) ? buf0[tid+1] : 0.0f;
            buf1[tid] = 0.25f * left + 0.5f * buf0[tid] + 0.25f * right;
        } else {
            // Read buf1, write buf0
            float left  = (tid > 0) ? buf1[tid-1] : 0.0f;
            float right = (tid < 255) ? buf1[tid+1] : 0.0f;
            buf0[tid] = 0.25f * left + 0.5f * buf1[tid] + 0.25f * right;
        }
        __syncthreads();
    }

    if (gid < n) {
        out[gid] = (iters % 2 == 0) ? buf0[tid] : buf1[tid];
    }
}

// ------------------------------------------------------------------
// Grid-stride loop with atomicAdd histogram.

__global__ void grid_histogram(int *hist, int *data, int bins, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int b = data[i] % bins;
        if (b >= 0 && b < bins) atomicAdd(&hist[b], 1);
    }
}

// ------------------------------------------------------------------
// Complex expression with many temporaries (tests register pressure).

__global__ void complex_expr(float *out, float *a, float *b, float *c,
                               float *d, float *e, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float va = a[gid], vb = b[gid], vc = c[gid];
        float vd = d[gid], ve = e[gid];
        // Horner's method for a degree-4 polynomial in 5 variables
        float r = va * (vb * (vc * (vd * ve + va) + vb) + vc) + vd;
        r = r * r - (va + vb + vc + vd + ve);
        r = sqrtf(fabsf(r)) + 1.0f / (fabsf(r) + 1e-6f);
        out[gid] = r;
    }
}

// ------------------------------------------------------------------
// Recursive power function.

__device__ float power(float base, int exp) {
    if (exp == 0) return 1.0f;
    if (exp == 1) return base;
    float half = power(base, exp / 2);
    float result = half * half;
    if (exp % 2 == 1) result *= base;
    return result;
}

__global__ void power_kernel(float *out, float *bases, int *exps, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = power(bases[tid], exps[tid]);
}
