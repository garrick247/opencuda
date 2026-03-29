// Probe: if/else struct assignment — both branches assign different values
// Without PHI nodes, the last-parsed branch's value may incorrectly win.
// Tests both constant and register cases.

struct Vec2 { float x, y; };

__device__ Vec2 make_v2(float x, float y) {
    Vec2 r; r.x = x; r.y = y; return r;
}

__device__ Vec2 scale_v2(Vec2 v, float s) {
    Vec2 r; r.x = v.x * s; r.y = v.y * s; return r;
}

// if/else with constant-result inlines in both branches
// Expected: flag>0 → out={1,0}, else → out={0,1}
// Bug: if constants not tracked per-branch, might always emit else values
__global__ void if_else_const_struct(float *out, float flag) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec2 result;
        if (flag > 0.0f) {
            result = make_v2(1.0f, 0.0f);
        } else {
            result = make_v2(0.0f, 1.0f);
        }
        out[0] = result.x;
        out[1] = result.y;
    }
}

// if/else with register-result inlines in both branches
// Expected: flag>0 → out = {in[0], in[0]*2}, else → out = {in[1], in[1]*3}
__global__ void if_else_reg_struct(float *out, float *in, float flag) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec2 result;
        if (flag > 0.0f) {
            result = scale_v2(make_v2(in[0], in[0]), 2.0f);
        } else {
            result = scale_v2(make_v2(in[1], in[1]), 3.0f);
        }
        out[0] = result.x;
        out[1] = result.y;
    }
}

// loop with if/else struct update (one path updates, one resets)
// Expected when in[i] >= 0: sum accumulates in[i], count++
//           when in[i] <  0: sum = 0, count = 0
__global__ void loop_if_else_struct(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec2 s; s.x = 0.0f; s.y = 0.0f;
        for (int i = 0; i < n; i++) {
            if (in[i] >= 0.0f) {
                s = make_v2(s.x + in[i], s.y + 1.0f);
            } else {
                s = make_v2(0.0f, 0.0f);
            }
        }
        out[0] = s.x;
        out[1] = s.y;
    }
}
