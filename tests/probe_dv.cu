// Probe: Patterns involving C++ style initializers and constructors 
// - Struct initialization with designated initializers (C99/C++20)
// - C++ uniform initialization: Vec3 v = {1.0f, 2.0f, 3.0f}
// - struct with = {0} zero-initialization
// - Anonymous struct init in declaration

struct Point3D {
    float x, y, z;
};

__global__ void struct_init_patterns(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // C-style struct init
        Point3D p;
        p.x = (float)tid;
        p.y = (float)(tid * 2);
        p.z = (float)(tid * 3);
        out[tid] = p.x + p.y + p.z;
    }
}

// Struct passed from kernel parameter
__global__ void struct_from_param(float *out, Point3D origin, Point3D *points, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Point3D p = points[tid];
        float dx = p.x - origin.x;
        float dy = p.y - origin.y;
        float dz = p.z - origin.z;
        out[tid] = sqrtf(dx*dx + dy*dy + dz*dz);
    }
}
