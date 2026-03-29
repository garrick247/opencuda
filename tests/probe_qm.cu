// Probe: nested structs, struct-returning __device__ functions,
// struct field chains (a.b.c), struct copy via assignment.

// ------------------------------------------------------------------
// Nested struct: a 2D point inside a bounding box.

struct Point2D {
    float x, y;
};

struct BBox {
    Point2D lo;
    Point2D hi;
};

__device__ BBox g_bbox;

__global__ void set_bbox(float lx, float ly, float hx, float hy) {
    if (threadIdx.x == 0) {
        g_bbox.lo.x = lx;
        g_bbox.lo.y = ly;
        g_bbox.hi.x = hx;
        g_bbox.hi.y = hy;
    }
}

__global__ void query_bbox(float *out, float px, float py) {
    if (threadIdx.x == 0) {
        float in_x = (px >= g_bbox.lo.x && px <= g_bbox.hi.x) ? 1.0f : 0.0f;
        float in_y = (py >= g_bbox.lo.y && py <= g_bbox.hi.y) ? 1.0f : 0.0f;
        out[0] = in_x * in_y;
    }
}

// ------------------------------------------------------------------
// Struct-returning __device__ function.

struct MinMax {
    float mn;
    float mx;
};

__device__ MinMax compute_minmax(float a, float b) {
    MinMax r;
    r.mn = (a < b) ? a : b;
    r.mx = (a > b) ? a : b;
    return r;
}

__global__ void minmax_kernel(float *in, float *out_mn, float *out_mx, int n) {
    int tid = threadIdx.x;
    if (tid < n - 1) {
        MinMax mm = compute_minmax(in[tid], in[tid + 1]);
        out_mn[tid] = mm.mn;
        out_mx[tid] = mm.mx;
    }
}

// ------------------------------------------------------------------
// Struct copy via assignment.

struct Vec2 {
    float x, y;
};

__global__ void copy_vecs(Vec2 *dst, Vec2 *src, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 v;
        v.x = src[tid].x;
        v.y = src[tid].y;
        dst[tid].x = v.x;
        dst[tid].y = v.y;
    }
}

// ------------------------------------------------------------------
// __device__ function that takes a struct arg and returns a scalar.

struct Triangle {
    float ax, ay, bx, by, cx, cy;
};

__device__ float triangle_area(Triangle t) {
    // Shoelace formula
    return 0.5f * ((t.bx - t.ax) * (t.cy - t.ay) -
                   (t.cx - t.ax) * (t.by - t.ay));
}

__global__ void areas_kernel(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 6;
        Triangle t;
        t.ax = data[base + 0]; t.ay = data[base + 1];
        t.bx = data[base + 2]; t.by = data[base + 3];
        t.cx = data[base + 4]; t.cy = data[base + 5];
        out[tid] = triangle_area(t);
    }
}

// ------------------------------------------------------------------
// Nested struct array: array of BBoxes.

__device__ BBox g_tiles[8];

__global__ void fill_tiles(float *lo_x, float *lo_y, float *hi_x, float *hi_y, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        g_tiles[tid].lo.x = lo_x[tid];
        g_tiles[tid].lo.y = lo_y[tid];
        g_tiles[tid].hi.x = hi_x[tid];
        g_tiles[tid].hi.y = hi_y[tid];
    }
}

__global__ void count_in_tiles(float *out, float px, float py, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        float inside = (px >= g_tiles[tid].lo.x && px <= g_tiles[tid].hi.x &&
                        py >= g_tiles[tid].lo.y && py <= g_tiles[tid].hi.y)
                       ? 1.0f : 0.0f;
        out[tid] = inside;
    }
}
