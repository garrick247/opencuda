// Probe: complex struct with bitfield-like manual packing,
// struct alignment padding awareness, struct array with pointer arithmetic

struct Particle {
    float x, y, z;    // position
    float vx, vy, vz; // velocity
    float mass;
    int type;
};

__device__ void integrate(Particle *p, float dt) {
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    p->z += p->vz * dt;
}

__device__ float kinetic_energy(Particle *p) {
    float v2 = p->vx * p->vx + p->vy * p->vy + p->vz * p->vz;
    return 0.5f * p->mass * v2;
}

__global__ void nbody_step(Particle *particles, float dt, float *ke_out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Particle *p = &particles[tid];
        // Euler integration
        integrate(p, dt);
        ke_out[tid] = kinetic_energy(p);
    }
}

// Sorting-related: compute sort key
__global__ void compute_sort_key(unsigned int *keys, Particle *particles, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Particle *p = &particles[tid];
        // Morton code-like: quantize position to 10-bit and interleave
        unsigned int xi = (unsigned int)(p->x * 1024.0f) & 0x3FF;
        unsigned int yi = (unsigned int)(p->y * 1024.0f) & 0x3FF;
        // Simple 20-bit key: 10 bits X | 10 bits Y
        keys[tid] = (xi << 10) | yi;
    }
}
