// Probe: struct in unrolled loop, struct mid-scope declaration,
//        device fn in while-loop with complex break

struct Vec2 { float x; float y; };
struct Grad { float gx; float gy; float mag; };

__device__ Grad compute_grad(float *field, int w, int r, int c) {
    float cx = field[r * w + c];
    float dx = field[r * w + c + 1] - field[r * w + c - 1];
    float dy = field[(r + 1) * w + c] - field[(r - 1) * w + c];
    Grad g;
    g.gx  = dx * 0.5f;
    g.gy  = dy * 0.5f;
    g.mag = g.gx * g.gx + g.gy * g.gy;
    return g;
}

// Struct used in unrolled loop (n=4 is compile-time, triggers unroller)
__global__ void unroll_grad_sum(float *out, float *field, int w) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec2 acc; acc.x = 0.0f; acc.y = 0.0f;
        // Inner unrollable loop: trip count = 4
        for (int i = 0; i < 4; i++) {
            Grad g = compute_grad(field, w, 1, i + 1);
            acc.x += g.gx;
            acc.y += g.gy;
        }
        out[0] = acc.x;
        out[1] = acc.y;
    }
}

// ---------------------------------------------------------------

struct State { float pos; float vel; float energy; };

__device__ State step_state(State s, float force, float dt) {
    s.vel += force * dt;
    s.pos += s.vel * dt;
    s.energy = 0.5f * s.vel * s.vel;
    return s;
}

// While-loop with complex break: stop when energy > limit or steps >= max
__global__ void simulate_state(float *out, float force, float dt,
                                float energy_limit, int max_steps) {
    int tid = threadIdx.x;
    if (tid == 0) {
        State s; s.pos = 0.0f; s.vel = 0.0f; s.energy = 0.0f;
        int steps = 0;
        while (steps < max_steps && s.energy < energy_limit) {
            s = step_state(s, force, dt);
            steps++;
        }
        out[0] = s.pos;
        out[1] = s.vel;
        out[2] = s.energy;
        out[3] = (float)steps;
    }
}

// ---------------------------------------------------------------

struct Pair2 { Vec2 a; Vec2 b; };  // nested struct

__device__ Vec2 midpoint(Vec2 a, Vec2 b) {
    Vec2 m;
    m.x = (a.x + b.x) * 0.5f;
    m.y = (a.y + b.y) * 0.5f;
    return m;
}

__device__ float dist2(Vec2 a, Vec2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Struct declared mid-scope inside if-block, nested struct usage
__global__ void midpoint_kernel(float *out, float *ax, float *ay,
                                  float *bx, float *by, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum_dist = 0.0f;
        float sum_mx = 0.0f;
        float sum_my = 0.0f;
        for (int i = 0; i < n; i++) {
            Vec2 pa; pa.x = ax[i]; pa.y = ay[i];
            Vec2 pb; pb.x = bx[i]; pb.y = by[i];
            Vec2 mid = midpoint(pa, pb);
            float d = dist2(pa, pb);
            sum_dist += d;
            sum_mx += mid.x;
            sum_my += mid.y;
        }
        out[0] = sum_dist;
        out[1] = sum_mx;
        out[2] = sum_my;
    }
}
