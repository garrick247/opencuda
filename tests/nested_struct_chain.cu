// Regression: nested struct member access chains (ptr[i].field.subfield)
// and struct assignment via lvalue address.
//
// Without fix 1 (nested struct field read):
//   p[tid].pos.x  → codegen tried to LoadInst a StructTy value (impossible)
//   Fix: when field type is StructTy, return address pointer (not load)
//
// Without fix 2 (chained lvalue assignment):
//   p[tid].pos.x += dt*vel;  → ParseError "expected SEMI, got DOT"
//   Fix: _parse_lvalue_or_expr now follows .field chains from ptr[i] base

typedef struct { float x; float y; } Vec2;
typedef struct { Vec2 pos; Vec2 vel; float mass; } Particle;

__global__ void particle_update(Particle *p, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Read nested fields
        float vx = p[tid].vel.x;
        float vy = p[tid].vel.y;
        // Write via chained lvalue (compound assignment)
        p[tid].pos.x += vx * dt;
        p[tid].pos.y += vy * dt;
        // Simple write via chained lvalue
        p[tid].mass = p[tid].mass * 0.99f;
    }
}

typedef struct { float x; float y; float z; } Vec3;
typedef struct { Vec3 origin; Vec3 dir; } Ray;

__global__ void ray_test(Ray *rays, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float ox = rays[tid].origin.x;
        float oy = rays[tid].origin.y;
        float oz = rays[tid].origin.z;
        float dx = rays[tid].dir.x;
        float dy = rays[tid].dir.y;
        float dz = rays[tid].dir.z;
        float t = 1.0f;
        // Ray at t: origin + t*dir
        out[tid * 3 + 0] = ox + t * dx;
        out[tid * 3 + 1] = oy + t * dy;
        out[tid * 3 + 2] = oz + t * dz;
    }
}
