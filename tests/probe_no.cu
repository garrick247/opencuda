// Probe: dual-condition loop (struct field used in loop condition),
//        struct + scalar chained inline (struct output fed to scalar inline),
//        while loop whose condition depends on a struct field updated by inline with ternary

// ------------------------------------------------------------------
// Dual-condition loop: i < n AND r.val < threshold.
// r.val is loop-carried and used both in the for_cond block (the second
// condition) and in the inline body.  Tests that r.val survives the
// back-edge liveness extension when it's last-used in the loop condition
// but last-defined in the ternary blocks that come after for_inc in flat order.

struct RunVal { float val; int cnt; };

__device__ RunVal accum_clamped(RunVal r, float v, float lo, float hi) {
    float c = (v > lo) ? v : lo;
    c = (c < hi) ? c : hi;
    r.val += c;
    r.cnt += 1;
    return r;
}

__global__ void dual_cond_accum(float *out, float *data, int n,
                                  float lo, float hi, float threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        RunVal r; r.val = 0.0f; r.cnt = 0;
        for (int i = 0; i < n && r.val < threshold; i++) {
            r = accum_clamped(r, data[i], lo, hi);
        }
        out[0] = r.val;
        out[1] = (float)r.cnt;
    }
}

// ------------------------------------------------------------------
// Struct + scalar chained inline:
// Loop iteration: make_vec2 → output struct V → blend(V, weight) → scalar.
// `weight` decays each iteration, so it's a loop-carried scalar that's
// passed as argument to blend's inline.  Tests that both struct fields
// from make_vec2 and the loop-carried weight reach blend's inline body.

struct Vec2 { float x; float y; };

__device__ Vec2 make_vec2(float a, float b) {
    Vec2 v;
    v.x = a;
    v.y = b;
    return v;
}

__device__ float blend_vec2(Vec2 v, float w) {
    return v.x * w + v.y * (1.0f - w);
}

__global__ void struct_scalar_chain(float *out, float *as, float *bs, int n,
                                     float init_w, float decay) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float total = 0.0f;
        float weight = init_w;
        for (int i = 0; i < n; i++) {
            Vec2 v = make_vec2(as[i], bs[i]);
            total += blend_vec2(v, weight);
            weight = weight * decay;
        }
        out[0] = total;
        out[1] = weight;
    }
}

// ------------------------------------------------------------------
// While loop with convergence: loop until pos^2 >= bound^2 or steps >= max.
// `p.pos` is used in the while condition AND updated by an inline that
// has a ternary clamp.  The inline merge block appears before the ternary
// blocks in flat order; p.pos must survive correctly into the while_cond.

struct PhysState { float pos; float vel; float energy; };

__device__ PhysState step_ps(PhysState p, float force, float dt) {
    p.vel += force * dt;
    p.pos += p.vel * dt;
    // clamp position
    p.pos = (p.pos > 100.0f) ? 100.0f : p.pos;
    p.energy = 0.5f * p.vel * p.vel;
    return p;
}

__global__ void physics_converge(float *out, float force, float dt,
                                   float bound, int max_steps) {
    int tid = threadIdx.x;
    if (tid == 0) {
        PhysState p; p.pos = 0.0f; p.vel = 0.0f; p.energy = 0.0f;
        int steps = 0;
        while (steps < max_steps && p.pos * p.pos < bound * bound) {
            p = step_ps(p, force, dt);
            steps++;
        }
        out[0] = p.pos;
        out[1] = p.vel;
        out[2] = p.energy;
        out[3] = (float)steps;
    }
}
