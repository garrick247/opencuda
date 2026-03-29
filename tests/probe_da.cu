// Probe: More unusual C/C++ syntax forms
// - Compound literal / designated initializer (should fail gracefully or parse)
// - Array of function pointers (skip/fail gracefully)
// - VLA (variable-length array) — not supported in CUDA but should fail gracefully
// - Initializer list: int arr[] = {1, 2, 3}  (unknown size from initializer)

// Struct with default-initialize pattern
struct Bbox {
    float x0, y0, x1, y1;
    float score;
    int label;
};

__device__ Bbox make_bbox(float x0, float y0, float x1, float y1, float score, int label) {
    Bbox b;
    b.x0 = x0; b.y0 = y0;
    b.x1 = x1; b.y1 = y1;
    b.score = score;
    b.label = label;
    return b;
}

__device__ float bbox_area(Bbox b) {
    float w = b.x1 - b.x0;
    float h = b.y1 - b.y0;
    return (w > 0.0f && h > 0.0f) ? w * h : 0.0f;
}

__global__ void compute_areas(float *out, float *x0, float *y0,
                               float *x1, float *y1, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Bbox b = make_bbox(x0[tid], y0[tid], x1[tid], y1[tid], 1.0f, 0);
        out[tid] = bbox_area(b);
    }
}
