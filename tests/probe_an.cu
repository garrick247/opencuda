// Probe: complex real-world pattern — multiple device functions calling each other,
//        struct passing by value vs by pointer,
//        device function with multiple params of mixed types

struct Ray {
    float ox, oy, oz;
    float dx, dy, dz;
};

struct Hit {
    float t;
    int id;
};

__device__ float dot3f(float ax, float ay, float az,
                       float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ Hit ray_sphere_intersect(Ray r, float cx, float cy, float cz, float radius, int id) {
    Hit h;
    h.t = -1.0f;
    h.id = -1;
    float ocx = r.ox - cx;
    float ocy = r.oy - cy;
    float ocz = r.oz - cz;
    float a = dot3f(r.dx, r.dy, r.dz, r.dx, r.dy, r.dz);
    float b = 2.0f * dot3f(ocx, ocy, ocz, r.dx, r.dy, r.dz);
    float c = dot3f(ocx, ocy, ocz, ocx, ocy, ocz) - radius * radius;
    float disc = b * b - 4.0f * a * c;
    if (disc >= 0.0f) {
        h.t = (-b - disc * 0.5f) / (2.0f * a);
        h.id = id;
    }
    return h;
}

__global__ void ray_trace(float *out, Ray *rays, float *spheres, int n_rays, int n_spheres) {
    int tid = threadIdx.x;
    if (tid < n_rays) {
        Ray r = rays[tid];
        float closest_t = 1e30f;
        int closest_id = -1;
        for (int i = 0; i < n_spheres; i++) {
            int base = i * 4;
            Hit h = ray_sphere_intersect(r, spheres[base], spheres[base+1],
                                         spheres[base+2], spheres[base+3], i);
            if (h.t > 0.0f && h.t < closest_t) {
                closest_t = h.t;
                closest_id = h.id;
            }
        }
        out[tid] = (float)closest_id + closest_t * 0.001f;
    }
}
