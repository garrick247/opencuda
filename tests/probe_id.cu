// Probe: complex address expressions — multi-level struct field offset,
// struct pointer arithmetic, mixed struct+array access patterns

struct Vec3 {
    float x, y, z;
};

struct AABB {
    Vec3 lo;
    Vec3 hi;
};

struct Triangle {
    Vec3 v0, v1, v2;
    int material_id;
};

__device__ float dot3(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Access nested struct fields via pointer
__global__ void compute_lengths(float *out, Vec3 *vecs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v = vecs[tid];
        out[tid] = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    }
}

// Access AABB lo/hi fields
__global__ void aabb_center(Vec3 *out, AABB *boxes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid].x = (boxes[tid].lo.x + boxes[tid].hi.x) * 0.5f;
        out[tid].y = (boxes[tid].lo.y + boxes[tid].hi.y) * 0.5f;
        out[tid].z = (boxes[tid].lo.z + boxes[tid].hi.z) * 0.5f;
    }
}

// Triangle material access
__global__ void filter_by_material(int *out, Triangle *tris, int mat, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (tris[tid].material_id == mat) ? 1 : 0;
    }
}
