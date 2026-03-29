// Probe: C++ style struct with constructor (no-op in CUDA device code)
// - Constructor body should be parsed but not emit meaningful IR
// - Destructor (same)
// - Member function call: v.normalize()

class Vec3 {
public:
    float x, y, z;
};

__device__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ Vec3 normalize(Vec3 v) {
    float len = length(v);
    Vec3 result;
    result.x = v.x / len;
    result.y = v.y / len;
    result.z = v.z / len;
    return result;
}

__global__ void lighting(float *out, Vec3 *normals, Vec3 light_dir, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 n_hat = normalize(normals[tid]);
        Vec3 l_hat = normalize(light_dir);
        float intensity = dot(n_hat, l_hat);
        out[tid] = (intensity > 0.0f) ? intensity : 0.0f;
    }
}
