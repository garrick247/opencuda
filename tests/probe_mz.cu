// Probe: inline-local variable (pre-branch computation) used by caller
// - classify() computes d=x*x+y*y before any branch, stores to r.dist
// - caller READS p.dist — must survive DCE
// - Also: two structs, one field used, one not, to verify selective DCE

struct Point  { int q; float dist; };
struct TagDist { float dist; int tag; };

// Local var 'd' computed BEFORE branch, stored in return field used by caller
__device__ Point classify(float x, float y) {
    float d = x*x + y*y;
    Point r;
    if (x >= 0.0f) r.q = 0; else r.q = 1;
    r.dist = d;
    return r;
}

// Local var computed BEFORE branch, caller uses BOTH fields
__device__ TagDist make_tag_dist(float v, int tag_base) {
    float d = v * v + 1.0f;
    TagDist r;
    r.dist = d;
    r.tag  = tag_base + (v >= 0.0f ? 0 : 1);
    return r;
}

// Caller reads p.dist — must not be DCE'd inside inline
__global__ void sum_distances(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float total = 0.0f;
        int q0 = 0;
        for (int i = 0; i < n; i++) {
            Point p = classify(xs[i], ys[i]);
            total += p.dist;      // USE dist — would be 0 if DCE'd
            if (p.q == 0) q0++;
        }
        out[0] = total;
        out[1] = (float)q0;
    }
}

// Caller reads BOTH fields — both must survive
__global__ void sum_tag_dist(float *out, float *vs, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float d_sum = 0.0f;
        int t_sum = 0;
        for (int i = 0; i < n; i++) {
            TagDist td = make_tag_dist(vs[i], i);
            d_sum += td.dist;
            t_sum += td.tag;
        }
        out[0] = d_sum;
        out[1] = (float)t_sum;
    }
}
