// Probe: Complex struct lifecycle — physics simulation, animation, particle systems
// - Struct with mixed float/int fields updated in loop via inline function
// - Struct param modified inside function body (field mutation before return)
// - Struct with conditional early return where param fields were mutated
// - Nested inline calls (inner result passed to outer)
// - Struct from multi-return fn used in switch

struct AnimState { float pos, vel; int frame, active; };
struct Particle { float x, y, z, vx, vy, vz, mass; int alive; };
struct Vec3 { float x, y, z; };

// Mixed-type struct with early return on inactive
__device__ AnimState advance(AnimState s, float dt) {
    if (!s.active) return s;
    s.pos += s.vel * dt;
    s.vel *= 0.99f;
    s.frame++;
    if (s.frame > 100) { s.active = 0; }
    return s;
}

__global__ void animate(float *out_pos, int *out_frame,
                        float *in_pos, float *in_vel, int *in_active,
                        float dt, int steps, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        AnimState s;
        s.pos = in_pos[tid]; s.vel = in_vel[tid];
        s.frame = 0; s.active = in_active[tid];
        for (int i = 0; i < steps; i++) {
            s = advance(s, dt);
        }
        out_pos[tid]   = s.pos;
        out_frame[tid] = s.frame;
    }
}

// Nested inline: advance_bounded calls advance internally
__device__ AnimState advance_bounded(AnimState s, float dt, float bound) {
    if (!s.active) return s;
    AnimState next = advance(s, dt);
    if (next.pos < -bound || next.pos > bound) {
        AnimState stopped = next;
        stopped.active = 0;
        return stopped;
    }
    return next;
}

__global__ void animate_bounded(float *out_pos, int *out_frame,
                                 float *in_pos, float *in_vel, int *in_active,
                                 float dt, float bound, int steps, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        AnimState s;
        s.pos = in_pos[tid]; s.vel = in_vel[tid];
        s.frame = 0; s.active = in_active[tid];
        for (int i = 0; i < steps; i++) {
            s = advance_bounded(s, dt, bound);
            if (!s.active) break;
        }
        out_pos[tid]   = s.pos;
        out_frame[tid] = s.frame;
    }
}

// Particle update with 7-field struct and conditional early return
__device__ Particle update_particle(Particle p, float dt, float gx, float gy, float gz) {
    if (!p.alive) return p;
    p.vx += gx * dt; p.vy += gy * dt; p.vz += gz * dt;
    p.x  += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
    if (p.y < 0.0f) { p.y = 0.0f; p.vy = -p.vy * 0.8f; }
    return p;
}

__global__ void simulate_particles(float *out, float *in_pos, float *in_vel,
                                    float *masses, int *alive,
                                    float dt, float gx, float gy, float gz,
                                    int steps, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Particle p;
        p.x=in_pos[tid*3]; p.y=in_pos[tid*3+1]; p.z=in_pos[tid*3+2];
        p.vx=in_vel[tid*3]; p.vy=in_vel[tid*3+1]; p.vz=in_vel[tid*3+2];
        p.mass=masses[tid]; p.alive=alive[tid];
        for (int s = 0; s < steps; s++) {
            p = update_particle(p, dt, gx, gy, gz);
        }
        out[tid*3]=p.x; out[tid*3+1]=p.y; out[tid*3+2]=p.z;
    }
}

// Cross product device fn returning struct, called in loop
__device__ Vec3 cross(Vec3 a, Vec3 b) {
    Vec3 r;
    r.x = a.y*b.z - a.z*b.y;
    r.y = a.z*b.x - a.x*b.z;
    r.z = a.x*b.y - a.y*b.x;
    return r;
}

__device__ Vec3 normalize3(Vec3 v) {
    float len2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len2 < 1e-10f) {
        Vec3 z; z.x=0.0f; z.y=0.0f; z.z=1.0f; return z;
    }
    float inv = 1.0f / len2;
    Vec3 r; r.x=v.x*inv; r.y=v.y*inv; r.z=v.z*inv; return r;
}

__global__ void angular_momentum(float *out, float *pos, float *vel, float *mass, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec3 L; L.x=0.0f; L.y=0.0f; L.z=0.0f;
        for (int i = 0; i < n; i++) {
            Vec3 r; r.x=pos[i*3]; r.y=pos[i*3+1]; r.z=pos[i*3+2];
            Vec3 v; v.x=vel[i*3]; v.y=vel[i*3+1]; v.z=vel[i*3+2];
            Vec3 rxv = cross(r, v);
            float m = mass[i];
            L.x += m * rxv.x;
            L.y += m * rxv.y;
            L.z += m * rxv.z;
        }
        Vec3 Ln = normalize3(L);
        out[0]=Ln.x; out[1]=Ln.y; out[2]=Ln.z;
    }
}
