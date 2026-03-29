// Probe: compound patterns combining previously-fixed bug classes.
// Each kernel exercises 2+ of: nested struct, global array subscript,
// __device__ fn call chain, shared memory, loop-carried phi, atomics.

// ------------------------------------------------------------------
// Global struct array + __device__ fn + nested struct field chain.

struct Vec2 {
    float x, y;
};

struct AABB2 {
    Vec2 lo;
    Vec2 hi;
};

__device__ AABB2 g_boxes[4];

__device__ int aabb_contains(AABB2 *box, float px, float py) {
    return (px >= box->lo.x && px <= box->hi.x &&
            py >= box->lo.y && py <= box->hi.y) ? 1 : 0;
}

__global__ void classify_points(int *out, float *pts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float px = pts[tid * 2 + 0];
        float py = pts[tid * 2 + 1];
        int hit = -1;
        for (int b = 0; b < 4; b++) {
            if (aabb_contains(&g_boxes[b], px, py)) {
                hit = b;
                break;
            }
        }
        out[tid] = hit;
    }
}

// ------------------------------------------------------------------
// Shared struct array + nested field write + __syncthreads + read.

struct Edge {
    int src;
    int dst;
    float weight;
};

__global__ void shared_edges(float *out, int *src_arr, int *dst_arr,
                              float *w_arr, int n) {
    __shared__ Edge sedge[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        sedge[tid].src    = src_arr[tid];
        sedge[tid].dst    = dst_arr[tid];
        sedge[tid].weight = w_arr[tid];
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        // Access neighbor edge
        int nb = (tid + 1) % n;
        out[tid] = sedge[nb].weight - sedge[tid].weight;
    }
}

// ------------------------------------------------------------------
// Global struct with atomicAdd on nested field + loop.

struct BinStats {
    int  count;
    float sum;
};

__device__ BinStats g_bins2[8];

__global__ void bin_accumulate(float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        int bin = (int)(v * 7.0f);
        if (bin < 0) bin = 0;
        if (bin > 7) bin = 7;
        atomicAdd(&g_bins2[bin].count, 1);
        // Note: atomicAdd on float field of struct
        atomicAdd(&g_bins2[bin].sum, v);
    }
}

// ------------------------------------------------------------------
// Loop with both shared memory write and global memory write per iter.

__global__ void hybrid_write(float *gout, float *gin, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    // Write to shared memory
    smem[lane] = (tid < n) ? gin[tid] : 0.0f;
    __syncthreads();
    // Reduction: each thread accumulates from shared
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < 32 && i < n; i++) {
            acc += smem[i];
        }
        gout[0] = acc;
    }
}

// ------------------------------------------------------------------
// Nested struct array: loop-carried index into nested field.

struct MatRow {
    float data[4];
    int   row_id;
};

__device__ MatRow g_mat[4];

__global__ void mat_trace(float *out) {
    if (threadIdx.x == 0) {
        float trace = 0.0f;
        for (int i = 0; i < 4; i++) {
            trace += g_mat[i].data[i];
        }
        out[0] = trace;
    }
}
