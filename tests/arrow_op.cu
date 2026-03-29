// Regression: ptr->field — arrow operator for pointer-to-struct member access
// Without fix: ARROW token not handled in parser → _parse_stmt called recursively
//   on the expression after '->' → ParseError "expected SEMI, got ARROW '->'".
// Fix: _parse_postfix_expr handles ARROW (rvalue load); _parse_lvalue_or_expr
//   handles ARROW (lvalue address return for StoreInst).

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    Vec3 pos;
    Vec3 vel;
    float mass;
} Particle;

// Device function writing via output pointer (ptr->field lvalue)
__device__ void init_particle(Particle *p, float x, float y, float z, float m) {
    p->pos.x = x;
    p->pos.y = y;
    p->pos.z = z;
    p->vel.x = 0.0f;
    p->vel.y = 0.0f;
    p->vel.z = 0.0f;
    p->mass = m;
}

// Kernel reading and writing struct pointer fields
__global__ void integrate(Particle *particles, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Read via -> (rvalue)
        float px = particles[tid].pos.x;
        float py = particles[tid].pos.y;
        float pz = particles[tid].pos.z;
        float vx = particles[tid].vel.x;
        float vy = particles[tid].vel.y;
        float vz = particles[tid].vel.z;

        // Write via -> (lvalue)
        particles[tid].pos.x = px + vx * dt;
        particles[tid].pos.y = py + vy * dt;
        particles[tid].pos.z = pz + vz * dt;
    }
}

// Simple direct -> read and write
__global__ void vec3_scale(Vec3 *vecs, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = vecs[tid].x;
        float y = vecs[tid].y;
        float z = vecs[tid].z;
        vecs[tid].x = x * scale;
        vecs[tid].y = y * scale;
        vecs[tid].z = z * scale;
    }
}
