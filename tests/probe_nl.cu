// Probe: float-register inline-merge liveness (same bug as probe_nh but float),
//        nested if-in-loop with struct carry, multi-field sequential ternary inline

struct FRange { float lo; float hi; float step; };
struct Clip { float lo; float hi; };  // both float, both from ternaries

// Two float ternaries: lo = max(a, bound_lo), hi = min(b, bound_hi)
// Tests that float variant of the inline-merge liveness fix works
__device__ Clip clip_range(float a, float b, float bound_lo, float bound_hi) {
    Clip c;
    c.lo = (a > bound_lo) ? a : bound_lo;
    c.hi = (b < bound_hi) ? b : bound_hi;
    return c;
}

__global__ void clip_accumulate(float *out, float *as, float *bs, int n,
                                  float lo_bound, float hi_bound) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum_lo = 0.0f, sum_hi = 0.0f;
        for (int i = 0; i < n; i++) {
            Clip c = clip_range(as[i], bs[i], lo_bound, hi_bound);
            sum_lo += c.lo;
            sum_hi += c.hi;
        }
        out[0] = sum_lo;
        out[1] = sum_hi;
    }
}

// ---------------------------------------------------------------

struct Slope { float k; float b; };  // y = k*x + b

// Three float ternaries in sequence: k, b, and a clamp
__device__ Slope fit_slope(float x0, float y0, float x1, float y1, float kmax) {
    Slope s;
    float dx = x1 - x0;
    s.k = (dx != 0.0f) ? (y1 - y0) / dx : 0.0f;
    s.b = y0 - s.k * x0;
    s.k = (s.k > kmax) ? kmax : s.k;  // clamp
    return s;
}

__global__ void slope_sum(float *out, float *xs, float *ys, int n, float kmax) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum_k = 0.0f, sum_b = 0.0f;
        for (int i = 0; i < n - 1; i++) {
            Slope s = fit_slope(xs[i], ys[i], xs[i+1], ys[i+1], kmax);
            sum_k += s.k;
            sum_b += s.b;
        }
        out[0] = sum_k;
        out[1] = sum_b;
    }
}

// ---------------------------------------------------------------

struct Box { float x; float y; float w; float h; };

__device__ float box_area(Box b) {
    return b.w * b.h;
}

__device__ Box box_intersect(Box a, Box b) {
    Box r;
    float x0 = (a.x > b.x) ? a.x : b.x;
    float y0 = (a.y > b.y) ? a.y : b.y;
    float x1_a = a.x + a.w;
    float x1_b = b.x + b.w;
    float x1 = (x1_a < x1_b) ? x1_a : x1_b;
    float y1_a = a.y + a.h;
    float y1_b = b.y + b.h;
    float y1 = (y1_a < y1_b) ? y1_a : y1_b;
    r.x = x0; r.y = y0;
    r.w = (x1 > x0) ? (x1 - x0) : 0.0f;
    r.h = (y1 > y0) ? (y1 - y0) : 0.0f;
    return r;
}

// Many sequential ternaries in one inline (6 ternaries for box_intersect)
__global__ void intersect_area_sum(float *out, float *ax, float *ay,
                                     float *aw, float *ah,
                                     float *bx, float *by,
                                     float *bw, float *bh, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            Box a; a.x = ax[i]; a.y = ay[i]; a.w = aw[i]; a.h = ah[i];
            Box b; b.x = bx[i]; b.y = by[i]; b.w = bw[i]; b.h = bh[i];
            Box isect = box_intersect(a, b);
            total += box_area(isect);
        }
        out[0] = total;
    }
}
