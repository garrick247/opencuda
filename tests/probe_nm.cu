// Probe: nested loop with outer variable used in inner-loop inline ternary,
//        dual sequential struct-returning inlines per iteration,
//        4-field struct with sequential if-block (non-ternary) updates

// ------------------------------------------------------------------
// Nested loop: outer loop computes a running max; inner loop uses
// that max as the clamp bound.  outer_max must survive the inner
// back-edge even though it's last "used" (flat-order) in the ternary
// sub-blocks that appear after inner_for_inc in the flat list.

struct ClampResult { float val; int clamped; };

__device__ ClampResult clamp_to(float v, float bound) {
    ClampResult r;
    r.val    = (v > bound) ? bound : v;
    r.clamped = (v > bound) ? 1 : 0;
    return r;
}

__global__ void nested_clamp_accum(float *out, float *data, int rows, int cols,
                                    float init_bound) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float outer_max  = init_bound;
        float total_val  = 0.0f;
        int   total_clamp = 0;
        for (int r = 0; r < rows; r++) {
            float row_sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float v = data[r * cols + c];
                // outer_max used as bound inside inner-loop inline (has ternaries)
                ClampResult cr = clamp_to(v, outer_max);
                row_sum += cr.val;
                total_clamp += cr.clamped;
            }
            total_val += row_sum;
            // outer_max grows: use row_sum as updated bound
            if (row_sum > outer_max) outer_max = row_sum;
        }
        out[0] = total_val;
        out[1] = (float)total_clamp;
        out[2] = outer_max;
    }
}

// ------------------------------------------------------------------
// Dual sequential struct inlines: two calls to the same device function
// in the same loop body.  Both produce struct returns, so two separate
// inline merge blocks are chained.  Tests that both writebacks land in
// distinct registers and neither clobbers the other.

struct Acc2 { float sum; float sumsq; };

__device__ Acc2 acc_update(Acc2 a, float v) {
    a.sum   += v;
    a.sumsq += v * v;
    return a;
}

__global__ void dual_acc_sum(float *out, float *pos, float *neg, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc2 pa; pa.sum = 0.0f; pa.sumsq = 0.0f;
        Acc2 na; na.sum = 0.0f; na.sumsq = 0.0f;
        for (int i = 0; i < n; i++) {
            pa = acc_update(pa, pos[i]);
            na = acc_update(na, neg[i]);
        }
        out[0] = pa.sum;
        out[1] = pa.sumsq;
        out[2] = na.sum;
        out[3] = na.sumsq;
    }
}

// ------------------------------------------------------------------
// 4-field struct with two sequential if-blocks (non-ternary) that
// conditionally modify the cur field.  Tests that all four fields
// survive through the conditional blocks and are correctly written back.

struct BoundedStep { float lo; float hi; float step; float cur; };

__device__ BoundedStep bs_advance(BoundedStep s) {
    s.cur += s.step;
    if (s.cur > s.hi) s.cur = s.hi;
    if (s.cur < s.lo) s.cur = s.lo;
    return s;
}

__global__ void bounded_walk(float *out, int n,
                              float lo, float hi, float step, float start) {
    int tid = threadIdx.x;
    if (tid == 0) {
        BoundedStep s; s.lo = lo; s.hi = hi; s.step = step; s.cur = start;
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            s = bs_advance(s);
            total += s.cur;
        }
        out[0] = total;
        out[1] = s.cur;
        out[2] = s.lo;
        out[3] = s.hi;
    }
}
