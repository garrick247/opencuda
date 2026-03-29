// Probe: Real production-quality kernel patterns from well-known CUDA samples
// - Naive N-body simulation (O(N²) force computation)
// - k-NN distance computation
// - Simple hash table insertion

struct Body {
    float x, y, z;   // position
    float vx, vy, vz; // velocity
    float mass;
};

__device__ void body_force(Body a, Body b, float *fx, float *fy, float *fz,
                            float eps) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    float dist_sq = dx*dx + dy*dy + dz*dz + eps;
    float inv_dist3 = rsqrtf(dist_sq) / dist_sq;
    float f = b.mass * inv_dist3;
    *fx += f * dx;
    *fy += f * dy;
    *fz += f * dz;
}

__global__ void nbody_update(Body *bodies, Body *out_bodies, int n, float dt, float eps) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    
    Body a = bodies[tid];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    
    for (int j = 0; j < n; j++) {
        if (j != tid) {
            body_force(a, bodies[j], &fx, &fy, &fz, eps);
        }
    }
    
    out_bodies[tid].vx = a.vx + fx * dt;
    out_bodies[tid].vy = a.vy + fy * dt;
    out_bodies[tid].vz = a.vz + fz * dt;
    out_bodies[tid].x  = a.x  + out_bodies[tid].vx * dt;
    out_bodies[tid].y  = a.y  + out_bodies[tid].vy * dt;
    out_bodies[tid].z  = a.z  + out_bodies[tid].vz * dt;
    out_bodies[tid].mass = a.mass;
}
