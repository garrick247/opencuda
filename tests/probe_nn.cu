// Probe: nested ternary inline (device fn calling device fn with ternaries),
//        two-pass loop sharing a scalar accumulator across loop boundaries,
//        5-field struct updated by inline

// ------------------------------------------------------------------
// Nested inline: norm_range calls safe_ratio twice; safe_ratio has a ternary.
// Two levels of inline merge blocks, each placed before its ternary sub-blocks.

struct RangeF { float lo; float hi; };

__device__ float safe_ratio(float num, float den, float dflt) {
    return (den != 0.0f) ? num / den : dflt;
}

__device__ RangeF norm_range(float a, float b, float scale, float dflt) {
    RangeF r;
    r.lo = safe_ratio(a, scale, dflt);
    r.hi = safe_ratio(b, scale, dflt);
    return r;
}

__global__ void nested_norm_accum(float *out, float *as, float *bs, int n,
                                   float scale, float dflt) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum_lo = 0.0f, sum_hi = 0.0f;
        for (int i = 0; i < n; i++) {
            RangeF r = norm_range(as[i], bs[i], scale, dflt);
            sum_lo += r.lo;
            sum_hi += r.hi;
        }
        out[0] = sum_lo;
        out[1] = sum_hi;
    }
}

// ------------------------------------------------------------------
// Two-pass loop: pass1 computes the sum; pass2 uses that sum to compute
// weighted normalized values.  The `total` variable from pass1 must
// remain live across the inter-loop gap and through pass2's body.

__device__ float weighted_norm(float v, float w, float total) {
    return (total > 0.0f) ? (v / total) * w : 0.0f;
}

__global__ void two_pass_norm(float *out, float *vals, float *weights, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Pass 1: compute sum
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            total += vals[i];
        }
        // Pass 2: weighted normalized sum — total used as argument to inline ternary
        float result = 0.0f;
        for (int i = 0; i < n; i++) {
            result += weighted_norm(vals[i], weights[i], total);
        }
        out[0] = total;
        out[1] = result;
    }
}

// ------------------------------------------------------------------
// 5-field struct: an affine transform state (scale, offset, bias, x, y).
// inline apply_transform modifies x and y each iteration.

struct AffineState { float scale; float offset; float bias; float x; float y; };

__device__ AffineState apply_transform(AffineState s) {
    s.x = s.scale * s.x + s.offset;
    s.y = s.scale * s.y + s.bias;
    return s;
}

__global__ void affine_walk(float *out, int n,
                             float scale, float offset, float bias,
                             float x0, float y0) {
    int tid = threadIdx.x;
    if (tid == 0) {
        AffineState s;
        s.scale  = scale;
        s.offset = offset;
        s.bias   = bias;
        s.x      = x0;
        s.y      = y0;
        float sum_x = 0.0f, sum_y = 0.0f;
        for (int i = 0; i < n; i++) {
            s = apply_transform(s);
            sum_x += s.x;
            sum_y += s.y;
        }
        out[0] = sum_x;
        out[1] = sum_y;
        out[2] = s.x;
        out[3] = s.y;
    }
}
