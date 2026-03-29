// Probe: arrow operator chains, pointer-to-pointer, struct arrays
// accessed via both [] and pointer arithmetic, and mixed struct field writes.

struct Vec3 { float x, y, z; };
struct Particle { struct Vec3 pos; struct Vec3 vel; float mass; };

// ------------------------------------------------------------------
// Arrow operator on pointer-to-struct.

__global__ void arrow_basic(float *out, struct Vec3 *vecs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec3 *v = &vecs[tid];
        out[tid] = v->x + v->y + v->z;
    }
}

// ------------------------------------------------------------------
// Arrow on nested struct (Particle has Vec3 pos and Vec3 vel).

__global__ void nested_arrow(float *out, struct Particle *parts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Particle *p = parts + tid;
        float dx = p->vel.x;
        float dy = p->vel.y;
        float dz = p->vel.z;
        out[tid] = dx * dx + dy * dy + dz * dz;  // speed squared
    }
}

// ------------------------------------------------------------------
// Update struct fields via arrow.

__global__ void euler_step(struct Particle *parts, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Particle *p = parts + tid;
        p->pos.x += p->vel.x * dt;
        p->pos.y += p->vel.y * dt;
        p->pos.z += p->vel.z * dt;
    }
}

// ------------------------------------------------------------------
// Struct pointer arithmetic: advance by 2.

__global__ void stride2_struct(float *out, struct Vec3 *vecs, int n) {
    int tid = threadIdx.x;
    if (tid * 2 < n) {
        struct Vec3 *even = vecs + tid * 2;
        struct Vec3 *odd  = even + 1;
        out[tid] = even->x + odd->x;
    }
}

// ------------------------------------------------------------------
// Array of pointers to structs (on device, via index trick).

__global__ void vec3_dot_kernel(float *out, struct Vec3 *a, struct Vec3 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float dot = a[tid].x * b[tid].x
                  + a[tid].y * b[tid].y
                  + a[tid].z * b[tid].z;
        out[tid] = dot;
    }
}

// ------------------------------------------------------------------
// Swap two fields via pointer.

__device__ void swap_xy(struct Vec3 *v) {
    float t = v->x;
    v->x = v->y;
    v->y = t;
}

__global__ void swap_fields(struct Vec3 *vecs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        swap_xy(&vecs[tid]);
    }
}

// ------------------------------------------------------------------
// Conditional struct field update via arrow.

__global__ void cond_field_update(struct Vec3 *vecs, float threshold, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec3 *v = vecs + tid;
        if (v->x > threshold) v->x = threshold;
        if (v->y > threshold) v->y = threshold;
        if (v->z > threshold) v->z = threshold;
    }
}

// ------------------------------------------------------------------
// Particle kinetic energy: 0.5 * mass * |vel|^2.

__global__ void kinetic_energy(float *out, struct Particle *parts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float vx = parts[tid].vel.x;
        float vy = parts[tid].vel.y;
        float vz = parts[tid].vel.z;
        float m  = parts[tid].mass;
        out[tid] = 0.5f * m * (vx*vx + vy*vy + vz*vz);
    }
}
