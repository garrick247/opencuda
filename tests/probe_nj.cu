// Probe: nested device fn calls (A calls B), struct with active field,
//        same struct passed to two different device fns in one iteration

struct Particle { float x; float y; float vx; float vy; int active; };

__device__ Particle advance(Particle p, float dt) {
    if (!p.active) return p;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    return p;
}

__device__ Particle bounce(Particle p, float wall) {
    if (!p.active) return p;
    if (p.x > wall)  { p.x = wall;  p.vx = -p.vx; }
    if (p.x < 0.0f)  { p.x = 0.0f; p.vx = -p.vx; }
    if (p.y > wall)  { p.y = wall;  p.vy = -p.vy; }
    if (p.y < 0.0f)  { p.y = 0.0f; p.vy = -p.vy; }
    return p;
}

// A calls B: advance_and_bounce calls advance then bounce
__device__ Particle advance_and_bounce(Particle p, float dt, float wall) {
    p = advance(p, dt);
    p = bounce(p, wall);
    return p;
}

__global__ void simulate(float *xs, float *ys, float *vxs, float *vys,
                          int *actives, int n, float dt, float wall) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            Particle p;
            p.x = xs[i]; p.y = ys[i];
            p.vx = vxs[i]; p.vy = vys[i];
            p.active = actives[i];
            p = advance_and_bounce(p, dt, wall);
            xs[i] = p.x; ys[i] = p.y;
            vxs[i] = p.vx; vys[i] = p.vy;
        }
    }
}

// ---------------------------------------------------------------

struct Seg { float a; float b; float c; };  // a*x + b = c

// Two different device fns take the same struct in one iteration
__device__ float eval_seg(Seg s, float x) {
    return s.a * x + s.b;
}

__device__ float seg_error(Seg s, float x, float y) {
    float pred = eval_seg(s, x);
    float diff = pred - y;
    return diff * diff;
}

__global__ void seg_fit_error(float *out, float *xs, float *ys, int n,
                               float a, float b) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Seg s; s.a = a; s.b = b; s.c = 0.0f;
        float total_err = 0.0f;
        float total_pred = 0.0f;
        for (int i = 0; i < n; i++) {
            // Same struct s passed to TWO different fns
            total_pred += eval_seg(s, xs[i]);
            total_err  += seg_error(s, xs[i], ys[i]);
        }
        out[0] = total_pred;
        out[1] = total_err;
    }
}

// ---------------------------------------------------------------

struct Running { float sum; float sum2; int cnt; float last; };

__device__ Running update_running(Running r, float x) {
    r.sum  += x;
    r.sum2 += x * x;
    r.cnt++;
    r.last = x;
    return r;
}

__device__ float running_mean(Running r) {
    return (r.cnt > 0) ? r.sum / (float)r.cnt : 0.0f;
}

__device__ float running_var(Running r) {
    if (r.cnt < 2) return 0.0f;
    float m = r.sum / (float)r.cnt;
    return r.sum2 / (float)r.cnt - m * m;
}

// Struct accumulator with multiple accessor device fns after the loop
__global__ void stats_output(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Running r; r.sum = 0.0f; r.sum2 = 0.0f; r.cnt = 0; r.last = 0.0f;
        for (int i = 0; i < n; i++) {
            r = update_running(r, in[i]);
        }
        out[0] = running_mean(r);
        out[1] = running_var(r);
        out[2] = r.last;
        out[3] = (float)r.cnt;
    }
}
