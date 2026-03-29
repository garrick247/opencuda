// Probe: Struct field comparison, IoU, NMS-style patterns
// - Struct passed by value to boolean-returning device function
// - Nested inline calls mixing scalar and struct
// - Struct used in all-branch-return device function called from loop

struct Box { float x, y, w, h; };

__device__ int boxes_overlap(Box a, Box b) {
    float ax2 = a.x + a.w;
    float ay2 = a.y + a.h;
    float bx2 = b.x + b.w;
    float by2 = b.y + b.h;
    return (a.x < bx2 && ax2 > b.x && a.y < by2 && ay2 > b.y);
}

__device__ float iou(Box a, Box b) {
    if (!boxes_overlap(a, b)) return 0.0f;
    float ix = (a.x + a.w < b.x + b.w ? a.x + a.w : b.x + b.w)
             - (a.x > b.x ? a.x : b.x);
    float iy = (a.y + a.h < b.y + b.h ? a.y + a.h : b.y + b.h)
             - (a.y > b.y ? a.y : b.y);
    if (ix <= 0.0f || iy <= 0.0f) return 0.0f;
    float inter = ix * iy;
    float ua = a.w * a.h;
    float ub = b.w * b.h;
    return inter / (ua + ub - inter);
}

__global__ void compute_iou(float *out, float *boxes, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i < n && j < n && i != j) {
        Box a; a.x=boxes[i*4]; a.y=boxes[i*4+1]; a.w=boxes[i*4+2]; a.h=boxes[i*4+3];
        Box b; b.x=boxes[j*4]; b.y=boxes[j*4+1]; b.w=boxes[j*4+2]; b.h=boxes[j*4+3];
        out[i*n+j] = iou(a, b);
    }
}

// All-branches-return struct fn called from loop
struct Interval { float lo, hi; };

__device__ Interval clamp_interval(Interval a, float lo, float hi) {
    if (a.lo >= hi) {
        Interval r; r.lo = hi; r.hi = hi;
        return r;
    }
    if (a.hi <= lo) {
        Interval r; r.lo = lo; r.hi = lo;
        return r;
    }
    Interval r;
    r.lo = a.lo < lo ? lo : a.lo;
    r.hi = a.hi > hi ? hi : a.hi;
    return r;
}

__global__ void clamp_intervals(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Interval iv;
        iv.lo = in[tid*2];
        iv.hi = in[tid*2+1];
        Interval clamped = clamp_interval(iv, 0.0f, 1.0f);
        out[tid*2]   = clamped.lo;
        out[tid*2+1] = clamped.hi;
    }
}

// Struct from multi-return fn used in switch
struct Quadrant { int q; float dist; };

__device__ Quadrant classify_point(float x, float y) {
    float d = x*x + y*y;
    if (x >= 0.0f && y >= 0.0f) {
        Quadrant r; r.q = 0; r.dist = d; return r;
    }
    if (x < 0.0f && y >= 0.0f) {
        Quadrant r; r.q = 1; r.dist = d; return r;
    }
    if (x < 0.0f && y < 0.0f) {
        Quadrant r; r.q = 2; r.dist = d; return r;
    }
    Quadrant r; r.q = 3; r.dist = d; return r;
}

__global__ void quadrant_histogram(int *hist, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int counts[4] = {0, 0, 0, 0};
        for (int i = 0; i < n; i++) {
            Quadrant q = classify_point(xs[i], ys[i]);
            if (q.q >= 0 && q.q < 4) {
                counts[q.q]++;
            }
        }
        hist[0]=counts[0]; hist[1]=counts[1]; hist[2]=counts[2]; hist[3]=counts[3];
    }
}
