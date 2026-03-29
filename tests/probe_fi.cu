// Probe: multiple return paths through complex control flow in __device__ functions
// Also: function returning struct through multiple branches

struct BBox {
    float xmin, xmax, ymin, ymax;
};

__device__ BBox compute_bbox(float *pts, int n) {
    if (n <= 0) {
        BBox empty;
        empty.xmin = 0.0f; empty.xmax = 0.0f;
        empty.ymin = 0.0f; empty.ymax = 0.0f;
        return empty;
    }
    BBox b;
    b.xmin = pts[0]; b.xmax = pts[0];
    b.ymin = pts[1]; b.ymax = pts[1];
    for (int i = 2; i < n * 2; i += 2) {
        float x = pts[i];
        float y = pts[i + 1];
        if (x < b.xmin) b.xmin = x;
        if (x > b.xmax) b.xmax = x;
        if (y < b.ymin) b.ymin = y;
        if (y > b.ymax) b.ymax = y;
    }
    return b;
}

__global__ void bbox_kernel(float *out, float *pts, int n_pts) {
    int tid = threadIdx.x;
    if (tid == 0) {
        BBox b = compute_bbox(pts, n_pts);
        out[0] = b.xmin;
        out[1] = b.xmax;
        out[2] = b.ymin;
        out[3] = b.ymax;
    }
}

// Device function with early return in loop
__device__ int first_nonzero(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] != 0) return i;
    }
    return -1;
}

__global__ void find_first(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = first_nonzero(in, n);
    }
}
