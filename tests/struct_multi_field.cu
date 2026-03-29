// Regression: comma-separated fields in struct: float x, y, z;
// Without fix: "float x, y, z;" inside struct → ParseError
//   "expected SEMI, got COMMA ','".
// Fix: _parse_struct_def loops on COMMA after each field name,
//   adding all names with the same type before consuming SEMI.

typedef struct {
    float x, y, z;
    int id;
} Particle;

typedef struct {
    int r, g, b, a;
} Color;

typedef struct {
    float u, v;
    float nx, ny, nz;
} Vertex;

__global__ void particle_sim(Particle *particles, float *forces, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fx = forces[tid * 3 + 0];
        float fy = forces[tid * 3 + 1];
        float fz = forces[tid * 3 + 2];
        particles[tid].x += fx * dt;
        particles[tid].y += fy * dt;
        particles[tid].z += fz * dt;
    }
}

__global__ void color_blend(Color *out, Color *a, Color *b, int n, float t) {
    int tid = threadIdx.x;
    if (tid < n) {
        int ti = (int)(t * 255.0f);
        out[tid].r = a[tid].r + (b[tid].r - a[tid].r) * ti / 255;
        out[tid].g = a[tid].g + (b[tid].g - a[tid].g) * ti / 255;
        out[tid].b = a[tid].b + (b[tid].b - a[tid].b) * ti / 255;
        out[tid].a = 255;
    }
}

__global__ void vertex_transform(Vertex *verts, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vertex v = verts[tid];
        // Transform texcoords and normal
        out[tid * 5 + 0] = v.u * 2.0f - 1.0f;
        out[tid * 5 + 1] = v.v * 2.0f - 1.0f;
        out[tid * 5 + 2] = v.nx;
        out[tid * 5 + 3] = v.ny;
        out[tid * 5 + 4] = v.nz;
    }
}
