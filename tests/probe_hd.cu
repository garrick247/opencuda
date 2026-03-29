// Probe: nested struct access (a.b.c), struct inside struct,
// postfix ++ on struct field, struct array element field mutation

struct Vec2 {
    float x;
    float y;
};

struct Particle {
    Vec2 pos;
    Vec2 vel;
    float mass;
};

// Nested struct field access: p.pos.x
__global__ void nested_struct(Particle *particles, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        particles[tid].pos.x += particles[tid].vel.x * dt;
        particles[tid].pos.y += particles[tid].vel.y * dt;
    }
}

// Postfix ++ on a struct field
struct Counter {
    int hits;
    int misses;
};

__global__ void field_inc(Counter *counters, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (flags[tid])
            counters[tid].hits++;
        else
            counters[tid].misses++;
    }
}

// Pointer to nested struct: access via ->
__device__ float mag_sq(Particle *p) {
    float vx = p->vel.x;
    float vy = p->vel.y;
    return vx * vx + vy * vy;
}

__global__ void speed_sq(float *out, Particle *particles, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = mag_sq(&particles[tid]);
    }
}
