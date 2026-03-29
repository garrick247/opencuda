// Probe: Struct field access in complex patterns
// - Struct inside conditional (assign only in one branch)
// - Array of structs iteration
// - Struct with pointer fields
// - Returning struct from __global__ via output pointer
// - Struct comparison (field by field)

struct Point {
    float x, y;
};

struct Rect {
    Point lo, hi;
};

// Write struct to output pointer (no struct return from __global__)
__global__ void struct_in_conditional(Point *out, float *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Point p;
        float v = vals[tid];
        if (v > 0.0f) {
            p.x = v;
            p.y = v * 2.0f;
        } else {
            p.x = -v;
            p.y = 0.0f;
        }
        out[tid].x = p.x;
        out[tid].y = p.y;
    }
}

// Array of structs: iterate and accumulate
__global__ void struct_array_sum(float *out, Point *pts, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sx = 0.0f, sy = 0.0f;
        for (int i = 0; i < n; i++) {
            sx += pts[i].x;
            sy += pts[i].y;
        }
        out[0] = sx;
        out[1] = sy;
    }
}

// Nested struct field access
__global__ void nested_struct_access(float *out, Rect *rects, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float w = rects[tid].hi.x - rects[tid].lo.x;
        float h = rects[tid].hi.y - rects[tid].lo.y;
        out[tid] = w * h;
    }
}

// Device function returning struct to caller
__device__ Point midpoint(Point a, Point b) {
    Point m;
    m.x = (a.x + b.x) * 0.5f;
    m.y = (a.y + b.y) * 0.5f;
    return m;
}

__global__ void struct_chain(float *out, Point *pts, int n) {
    int tid = threadIdx.x;
    if (tid + 1 < n) {
        Point a = pts[tid];
        Point b = pts[tid + 1];
        Point m = midpoint(a, b);
        out[tid] = m.x + m.y;
    }
}
