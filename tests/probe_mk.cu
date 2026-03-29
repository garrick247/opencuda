// Probe: Complex loop patterns with carried structs and early exits
// - Loop where struct is modified across iterations (not just scalars)
// - While loop condition calling device function that modifies state
// - Nested loops with break propagation through struct
// - Device function with early return in multiple branches
// - Loop with accumulation into struct fields

struct BBox {
    float x_min, x_max;
    float y_min, y_max;
};

__device__ BBox bbox_expand(BBox b, float px, float py) {
    if (px < b.x_min) b.x_min = px;
    if (px > b.x_max) b.x_max = px;
    if (py < b.y_min) b.y_min = py;
    if (py > b.y_max) b.y_max = py;
    return b;
}

__device__ float bbox_area(BBox b) {
    float w = b.x_max - b.x_min;
    float h = b.y_max - b.y_min;
    if (w < 0.0f || h < 0.0f) return 0.0f;
    return w * h;
}

// Struct accumulated across loop iterations via device function
__global__ void compute_bbox(float *xs, float *ys, float *area_out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        BBox b;
        b.x_min = xs[0]; b.x_max = xs[0];
        b.y_min = ys[0]; b.y_max = ys[0];
        for (int i = 1; i < n; i++) {
            b = bbox_expand(b, xs[i], ys[i]);
        }
        area_out[0] = bbox_area(b);
    }
}

struct MinMax {
    float mn;
    float mx;
    int idx_mn;
    int idx_mx;
};

__device__ MinMax find_minmax(float *arr, int n) {
    MinMax mm;
    mm.mn = arr[0];
    mm.mx = arr[0];
    mm.idx_mn = 0;
    mm.idx_mx = 0;
    for (int i = 1; i < n; i++) {
        float v = arr[i];
        if (v < mm.mn) { mm.mn = v; mm.idx_mn = i; }
        if (v > mm.mx) { mm.mx = v; mm.idx_mx = i; }
    }
    return mm;
}

__global__ void range_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        MinMax mm = find_minmax(in, n);
        out[0] = mm.mx - mm.mn;
        out[1] = (float)mm.idx_mn;
        out[2] = (float)mm.idx_mx;
    }
}

// Struct with early return in multiple branches (covers probe_fi/fu territory)
__device__ BBox merge_or_null(BBox a, BBox b, float overlap_thresh) {
    float dx = b.x_min - a.x_max;
    float dy = b.y_min - a.y_max;
    if (dx > overlap_thresh) {
        BBox empty;
        empty.x_min = 0.0f; empty.x_max = 0.0f;
        empty.y_min = 0.0f; empty.y_max = 0.0f;
        return empty;
    }
    BBox merged;
    merged.x_min = a.x_min < b.x_min ? a.x_min : b.x_min;
    merged.x_max = a.x_max > b.x_max ? a.x_max : b.x_max;
    merged.y_min = a.y_min < b.y_min ? a.y_min : b.y_min;
    merged.y_max = a.y_max > b.y_max ? a.y_max : b.y_max;
    return merged;
}

__global__ void bbox_merge_kernel(float *out, float *in, int n, float thresh) {
    int tid = threadIdx.x;
    if (tid == 0) {
        BBox a, b;
        a.x_min = in[0]; a.x_max = in[1]; a.y_min = in[2]; a.y_max = in[3];
        b.x_min = in[4]; b.x_max = in[5]; b.y_min = in[6]; b.y_max = in[7];
        BBox m = merge_or_null(a, b, thresh);
        out[0] = bbox_area(m);
    }
}
