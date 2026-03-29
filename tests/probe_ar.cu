// Probe: nested structs, struct assignment, struct member address-of
struct Vec3 {
    float x, y, z;
};

struct Particle {
    Vec3 pos;
    Vec3 vel;
    float mass;
};

__device__ Vec3 vec3_add(Vec3 a, Vec3 b) {
    Vec3 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    return r;
}

__device__ Vec3 vec3_scale(Vec3 v, float s) {
    Vec3 r;
    r.x = v.x * s;
    r.y = v.y * s;
    r.z = v.z * s;
    return r;
}

__global__ void integrate(Particle *particles, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Particle p = particles[tid];
        Vec3 dp = vec3_scale(p.vel, dt);
        p.pos = vec3_add(p.pos, dp);
        particles[tid].pos.x = p.pos.x;
        particles[tid].pos.y = p.pos.y;
        particles[tid].pos.z = p.pos.z;
    }
}
