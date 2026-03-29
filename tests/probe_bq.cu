// Probe: Constructor syntax, initializer lists, complex typedef patterns
// - int arr[] = {1, 2, 3}; (array init without size)
// - Multi-line string (should parse as one string)
// - Compound literal: (Type){...}
// - #include handling (should be skipped or handled)
// - Nested function calls with struct args

struct AABB {
    float min_x, min_y, max_x, max_y;
};

__device__ AABB make_aabb(float x0, float y0, float x1, float y1) {
    AABB b;
    b.min_x = x0; b.min_y = y0;
    b.max_x = x1; b.max_y = y1;
    return b;
}

__device__ int aabb_contains(AABB b, float x, float y) {
    return (x >= b.min_x && x <= b.max_x &&
            y >= b.min_y && y <= b.max_y) ? 1 : 0;
}

__global__ void point_in_box(int *out, float *xs, float *ys,
                              float x0, float y0, float x1, float y1, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        AABB box = make_aabb(x0, y0, x1, y1);
        out[tid] = aabb_contains(box, xs[tid], ys[tid]);
    }
}
