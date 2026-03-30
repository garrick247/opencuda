// Probe: stress test combining multiple features — recursive function
// called from grid-stride loop, shared memory reduction calling device
// function, complex nested struct with array member, multi-level pointer
// chain in recursive call, and large kernel with 15+ params.

// ------------------------------------------------------------------
// Recursive binary search.

__device__ int binary_search(int *arr, int target, int lo, int hi) {
    if (lo > hi) return -1;
    int mid = (lo + hi) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] < target) return binary_search(arr, target, mid + 1, hi);
    return binary_search(arr, target, lo, mid - 1);
}

__global__ void bsearch_kernel(int *out, int *sorted, int *queries,
                                  int arr_len, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = binary_search(sorted, queries[tid], 0, arr_len - 1);
}

// ------------------------------------------------------------------
// Grid-stride loop calling non-recursive device function.

__device__ float activation(float x) {
    // Mish activation: x * tanh(softplus(x)) ≈ x * tanh(ln(1+exp(x)))
    float sp = logf(1.0f + expf(x));
    // Approximate tanh via Pade
    float sp2 = sp * sp;
    float t = sp * (27.0f + sp2) / (27.0f + 9.0f * sp2);
    return x * t;
}

__global__ void mish_grid_stride(float *out, float *in, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = activation(in[i]);
    }
}

// ------------------------------------------------------------------
// Kernel with 16 parameters.

__global__ void many_params(float *out,
                               float *a, float *b, float *c, float *d,
                               float *e, float *f, float *g, float *h,
                               float w0, float w1, float w2, float w3,
                               float w4, float w5, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = a[tid]*w0 + b[tid]*w1 + c[tid]*w2 + d[tid]*w3
                  + e[tid]*w4 + f[tid]*w5 + g[tid] + h[tid];
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Complex reduction: sum, min, max, count in one pass.

struct Stats {
    float sum;
    float min_val;
    float max_val;
    int count;
};

__global__ void multi_reduce(float *out_sum, float *out_min, float *out_max,
                                int *out_count, float *in, float threshold, int n) {
    __shared__ float s_sum[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int   s_cnt[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float v = (gid < n) ? in[gid] : 0.0f;
    s_sum[tid] = v;
    s_min[tid] = (gid < n) ? v : 1e30f;
    s_max[tid] = (gid < n) ? v : -1e30f;
    s_cnt[tid] = (gid < n && v > threshold) ? 1 : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            s_cnt[tid] += s_cnt[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_sum[blockIdx.x]   = s_sum[0];
        out_min[blockIdx.x]   = s_min[0];
        out_max[blockIdx.x]   = s_max[0];
        out_count[blockIdx.x] = s_cnt[0];
    }
}

// ------------------------------------------------------------------
// Parallel k-means iteration: assign each point to nearest centroid.

__global__ void kmeans_assign(int *assignments, float *points_x, float *points_y,
                                 float *centroids_x, float *centroids_y,
                                 int K, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float px = points_x[gid], py = points_y[gid];
    float best_dist = 1e30f;
    int best_k = 0;
    for (int k = 0; k < K; k++) {
        float dx = px - centroids_x[k];
        float dy = py - centroids_y[k];
        float dist = dx*dx + dy*dy;
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }
    assignments[gid] = best_k;
}

// ------------------------------------------------------------------
// Euler integration for particle simulation.

__global__ void euler_step(float *px, float *py, float *pz,
                              float *vx, float *vy, float *vz,
                              float *ax, float *ay, float *az,
                              float dt, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    // v += a * dt
    vx[gid] += ax[gid] * dt;
    vy[gid] += ay[gid] * dt;
    vz[gid] += az[gid] * dt;
    // p += v * dt
    px[gid] += vx[gid] * dt;
    py[gid] += vy[gid] * dt;
    pz[gid] += vz[gid] * dt;
}
