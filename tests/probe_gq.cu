// Probe: C++ constructor-style initialization for structs (C++ aggregate init)
// Vec3 v = {1.0f, 2.0f, 3.0f} and Vec3 v{1.0f, 2.0f, 3.0f}

struct Vec3 {
    float x, y, z;
};

__device__ Vec3 normalize(Vec3 v) {
    float len_sq = v.x*v.x + v.y*v.y + v.z*v.z;
    float inv_len = rsqrtf(len_sq + 1e-8f);
    Vec3 r;
    r.x = v.x * inv_len;
    r.y = v.y * inv_len;
    r.z = v.z * inv_len;
    return r;
}

__global__ void vec3_brace_init(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 3;
        // C-style aggregate initializer
        Vec3 v = {in[base], in[base+1], in[base+2]};
        Vec3 u = normalize(v);
        out[base] = u.x;
        out[base+1] = u.y;
        out[base+2] = u.z;
    }
}

// Test that struct return from device function works correctly
// when assigned to a local struct variable (not just used inline)
__global__ void struct_return_assign(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid * 3;
        Vec3 v;
        v.x = in[base]; v.y = in[base+1]; v.z = in[base+2];
        Vec3 nrm = normalize(v);  // struct return
        Vec3 nrm2 = normalize(nrm);  // chain
        out[tid] = nrm2.x + nrm2.y + nrm2.z;
    }
}
