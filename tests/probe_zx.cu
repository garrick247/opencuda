// Probe: physics simulation patterns — Verlet integration, spring-damper,
// spatial hashing, collision detection AABB, fluid SPH density,
// electric field computation, and thermal diffusion.

// ------------------------------------------------------------------
// Verlet integration for cloth/soft body.

__global__ void verlet_step(float *px, float *py, float *pz,
                               float *ox, float *oy, float *oz,
                               float *fx, float *fy, float *fz,
                               float *inv_mass, float dt, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float im = inv_mass[gid];
    float ax = fx[gid] * im, ay = fy[gid] * im, az = fz[gid] * im;
    float nx = 2.0f * px[gid] - ox[gid] + ax * dt * dt;
    float ny = 2.0f * py[gid] - oy[gid] + ay * dt * dt;
    float nz = 2.0f * pz[gid] - oz[gid] + az * dt * dt;
    ox[gid] = px[gid]; oy[gid] = py[gid]; oz[gid] = pz[gid];
    px[gid] = nx; py[gid] = ny; pz[gid] = nz;
}

// ------------------------------------------------------------------
// Spring force: F = -k * (dist - rest_len) * dir.

__global__ void spring_force(float *fx, float *fy,
                                float *px, float *py,
                                int *spring_a, int *spring_b,
                                float *rest_len, float k, int n_springs) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_springs) return;
    int a = spring_a[gid], b = spring_b[gid];
    float dx = px[b] - px[a];
    float dy = py[b] - py[a];
    float dist = sqrtf(dx*dx + dy*dy);
    float stretch = dist - rest_len[gid];
    if (dist < 1e-8f) return;
    float force = k * stretch / dist;
    float ffx = force * dx, ffy = force * dy;
    atomicAdd(&fx[a], ffx);
    atomicAdd(&fy[a], ffy);
    atomicAdd(&fx[b], -ffx);
    atomicAdd(&fy[b], -ffy);
}

// ------------------------------------------------------------------
// AABB collision detection: check all pairs in a tile.

__global__ void aabb_check(int *collisions, int *count,
                              float *min_x, float *max_x,
                              float *min_y, float *max_y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n || i >= j) return;
    // AABB overlap test
    if (min_x[i] <= max_x[j] && max_x[i] >= min_x[j] &&
        min_y[i] <= max_y[j] && max_y[i] >= min_y[j]) {
        int pos = atomicAdd(count, 1);
        collisions[pos * 2]     = i;
        collisions[pos * 2 + 1] = j;
    }
}

// ------------------------------------------------------------------
// SPH density estimation.

__global__ void sph_density(float *density, float *px, float *py,
                               float *mass, float h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float rho = 0.0f;
    float h2 = h * h;
    for (int j = 0; j < n; j++) {
        float dx = px[i] - px[j];
        float dy = py[i] - py[j];
        float r2 = dx*dx + dy*dy;
        if (r2 < h2) {
            float q = 1.0f - sqrtf(r2) / h;
            rho += mass[j] * q * q * q;
        }
    }
    density[i] = rho;
}

// ------------------------------------------------------------------
// Electric field from point charges (Coulomb's law).

__global__ void electric_field(float *Ex, float *Ey,
                                  float *probe_x, float *probe_y,
                                  float *charge_x, float *charge_y,
                                  float *charge_q, float eps,
                                  int n_probes, int n_charges) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_probes) return;
    float prx = probe_x[gid], pry = probe_y[gid];
    float ex = 0.0f, ey = 0.0f;
    for (int j = 0; j < n_charges; j++) {
        float dx = prx - charge_x[j];
        float dy = pry - charge_y[j];
        float r2 = dx*dx + dy*dy + eps;
        float inv_r3 = rsqrtf(r2) / r2;
        float q = charge_q[j];
        ex += q * dx * inv_r3;
        ey += q * dy * inv_r3;
    }
    Ex[gid] = ex;
    Ey[gid] = ey;
}

// ------------------------------------------------------------------
// 2D thermal diffusion (explicit Euler).

__global__ void thermal_diffuse(float *next, float *curr,
                                   float alpha, float dt, float dx2,
                                   int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) return;
    float c = curr[y*W+x];
    float laplacian = curr[(y-1)*W+x] + curr[(y+1)*W+x]
                    + curr[y*W+(x-1)] + curr[y*W+(x+1)] - 4.0f*c;
    next[y*W+x] = c + alpha * dt * laplacian / dx2;
}
