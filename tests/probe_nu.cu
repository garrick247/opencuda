// Probe: cross-inline field dependency + nested if-inside-loop struct update
// Tests that a second inline sees the UPDATED fields from a first inline,
// and that both branches of an if inside a loop correctly writeback structs.

// ------------------------------------------------------------------
// Cross-inline dependency: func2 reads p.a which was just written by func1.
// After the chain, p.b must equal the NEW p.a (written by func1),
// not the original p.a from before the loop iteration.

struct Cascade { float a; float b; };

__device__ Cascade step_a(Cascade c, float v) {
    c.a += v;
    return c;
}

__device__ Cascade step_b(Cascade c) {
    c.b = c.a;   // b copies the CURRENT (possibly just-updated) a
    return c;
}

__global__ void cross_inline_dep(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Cascade c; c.a = 0.0f; c.b = 0.0f;
        for (int i = 0; i < n; i++) {
            c = step_a(c, data[i]);   // c.a += data[i]
            c = step_b(c);             // c.b = c.a  (new a!)
        }
        out[0] = c.a;
        out[1] = c.b;
    }
}

// ------------------------------------------------------------------
// Nested if-inside-loop: both branches call a struct-updating inline.
// Each branch calls a different function, both modifying the same field.
// Tests that both branches produce a correctly sequenced writeback and
// that the merge point sees the right register regardless of branch taken.

struct Counter { float pos_sum; float neg_sum; };

__device__ Counter add_pos(Counter c, float v) {
    c.pos_sum += v;
    return c;
}

__device__ Counter add_neg(Counter c, float v) {
    c.neg_sum += v;
    return c;
}

__global__ void branch_inline(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Counter c; c.pos_sum = 0.0f; c.neg_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            if (v >= 0.0f) {
                c = add_pos(c, v);
            } else {
                c = add_neg(c, v);
            }
        }
        out[0] = c.pos_sum;
        out[1] = c.neg_sum;
    }
}

// ------------------------------------------------------------------
// Struct with three fields where two inlines each update one field,
// and the third field is never touched (it must survive both writebacks).
// Tests that the unmodified field is correctly preserved through the
// two-inline chain.

struct Trio { float x; float y; float z; };

__device__ Trio update_x(Trio t, float v) {
    t.x += v;
    return t;
}

__device__ Trio update_y(Trio t, float v) {
    t.y += v;
    return t;
}

__global__ void preserve_z(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Trio t; t.x = 0.0f; t.y = 0.0f; t.z = 42.0f;
        for (int i = 0; i < n; i++) {
            t = update_x(t, xs[i]);
            t = update_y(t, ys[i]);
        }
        out[0] = t.x;
        out[1] = t.y;
        out[2] = t.z;   // must be 42.0f
    }
}
