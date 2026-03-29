// Probe: struct pointer arithmetic, -> chains on pointer-indexed arrays,
// address-of struct fields, and pointer-to-struct patterns.

struct Node { int val; int next_idx; };
struct Rect { int x, y, w, h; };
struct Color { unsigned char r, g, b, a; };

// ------------------------------------------------------------------
// Pointer-indexed struct array with -> access.

__global__ void node_walk(int *out, struct Node *nodes, int start, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Walk linked list for up to 8 steps
        int idx = start + tid;
        int sum = 0;
        for (int step = 0; step < 8 && idx >= 0 && idx < n; step++) {
            sum += nodes[idx].val;
            idx = nodes[idx].next_idx;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Struct array with pointer arithmetic.

__global__ void rect_area(int *out, struct Rect *rects, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Rect *r = rects + tid;
        out[tid] = r->w * r->h;
    }
}

// ------------------------------------------------------------------
// Arrow access on struct pointer param.

__global__ void rect_perimeter(int *out, struct Rect *rects, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = 2 * (rects[tid].w + rects[tid].h);
    }
}

// ------------------------------------------------------------------
// Struct containing another struct via flat encoding.

struct BBox { struct Rect outer; struct Rect inner; };

__global__ void bbox_area_diff(int *out, struct BBox *boxes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int outer_area = boxes[tid].outer.w * boxes[tid].outer.h;
        int inner_area = boxes[tid].inner.w * boxes[tid].inner.h;
        out[tid] = outer_area - inner_area;
    }
}

// ------------------------------------------------------------------
// Write to struct field via pointer.

__global__ void write_rect_fields(struct Rect *out, int *xs, int *ys, int *ws, int *hs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid].x = xs[tid];
        out[tid].y = ys[tid];
        out[tid].w = ws[tid];
        out[tid].h = hs[tid];
    }
}

// ------------------------------------------------------------------
// Color packing: 4 byte-sized fields packed into int.

__global__ void color_pack(unsigned int *out, struct Color *colors, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int r = (unsigned int)colors[tid].r;
        unsigned int g = (unsigned int)colors[tid].g;
        unsigned int b = (unsigned int)colors[tid].b;
        unsigned int a = (unsigned int)colors[tid].a;
        out[tid] = (a << 24) | (b << 16) | (g << 8) | r;
    }
}

// ------------------------------------------------------------------
// Struct passed by pointer to device function.

__device__ int rect_contains(struct Rect *r, int px, int py) {
    return (px >= r->x && px < r->x + r->w &&
            py >= r->y && py < r->y + r->h) ? 1 : 0;
}

__global__ void point_in_rect(int *out, struct Rect *rects, int *px, int *py, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = rect_contains(&rects[tid], px[tid], py[tid]);
    }
}

// ------------------------------------------------------------------
// Two different struct types in same kernel.

__global__ void mixed_structs(int *out, struct Rect *rects, struct Node *nodes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int area = rects[tid].w * rects[tid].h;
        int val  = nodes[tid].val;
        out[tid] = area + val;
    }
}
