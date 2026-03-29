// Probe: multiple inline calls per loop iteration + aliased field patterns
// Tests that return-value temporaries from two back-to-back inline calls
// in the same loop body do not collide, and that a field used as both
// input argument and output writeback target is handled correctly.

// ------------------------------------------------------------------
// Double inline per iteration: two struct-updating calls in one loop body.
// Each call updates a different pair of fields so there is no dependency
// between the two calls, but both share the same entry Values as sources.
// Tests that the second inline's return temporaries don't alias the first's.

struct Pair { float lo; float hi; };

__device__ Pair clamp_lo(Pair p, float v) {
    if (v < p.lo) p.lo = v;
    return p;
}

__device__ Pair clamp_hi(Pair p, float v) {
    if (v > p.hi) p.hi = v;
    return p;
}

__global__ void double_clamp(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Pair p; p.lo = data[0]; p.hi = data[0];
        for (int i = 1; i < n; i++) {
            float v = data[i];
            p = clamp_lo(p, v);
            p = clamp_hi(p, v);
        }
        out[0] = p.lo;
        out[1] = p.hi;
    }
}

// ------------------------------------------------------------------
// Field used as argument to same-struct inline: p.hi is passed as the
// threshold, and the inline also writes p.hi.  The read of p.hi for the
// argument must occur BEFORE the writeback of p.hi.

struct Thresh { float val; float thr; };

__device__ Thresh update_thresh(Thresh t, float v, float new_thr) {
    t.val  += v;
    t.thr   = new_thr;
    return t;
}

__global__ void field_as_arg(float *out, float *data, float *thresholds, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Thresh t; t.val = 0.0f; t.thr = thresholds[0];
        for (int i = 0; i < n; i++) {
            // t.thr is read here as new_thr, then overwritten inside the inline
            t = update_thresh(t, data[i], thresholds[i]);
        }
        out[0] = t.val;
        out[1] = t.thr;
    }
}

// ------------------------------------------------------------------
// Chained struct output → next call input: the result of one inline
// immediately feeds into the next inline in the same statement.
// Tests that the parser correctly handles chained struct assignments.

struct Acc { float sum; float wsum; };

__device__ Acc add_weighted(Acc a, float v, float w) {
    a.sum  += v;
    a.wsum += v * w;
    return a;
}

__global__ void chained_weight(float *out, float *vals, float *wts, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc a; a.sum = 0.0f; a.wsum = 0.0f;
        for (int i = 0; i < n; i++) {
            a = add_weighted(a, vals[i], wts[i]);
        }
        out[0] = a.sum;
        out[1] = a.wsum;
    }
}
