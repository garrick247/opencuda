// Probe: struct value assignment, copy, partial update
// Also: struct passed by value to __device__ function

struct Vec3 {
    float x, y, z;
};

__device__ Vec3 vec3_add(Vec3 a, Vec3 b) {
    Vec3 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    return r;
}

__device__ float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3 vec3_scale(Vec3 v, float s) {
    Vec3 r;
    r.x = v.x * s;
    r.y = v.y * s;
    r.z = v.z * s;
    return r;
}

__global__ void vec3_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 3;
        Vec3 va;
        va.x = a[base];
        va.y = a[base + 1];
        va.z = a[base + 2];

        Vec3 vb;
        vb.x = b[base];
        vb.y = b[base + 1];
        vb.z = b[base + 2];

        Vec3 sum = vec3_add(va, vb);
        float dot = vec3_dot(va, vb);
        Vec3 scaled = vec3_scale(sum, dot);

        out[tid] = scaled.x + scaled.y + scaled.z;
    }
}
