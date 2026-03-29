// Probe: deeply nested struct — struct containing struct containing struct
// Also: pointer to nested struct, access chain s.a.b.c

struct Color {
    float r, g, b, a;
};

struct Material {
    Color diffuse;
    Color specular;
    float shininess;
};

struct Object {
    Material mat;
    int id;
    float scale;
};

__device__ float luminance(Color c) {
    return 0.299f * c.r + 0.587f * c.g + 0.114f * c.b;
}

__global__ void shade_objects(float *out, Object *objs, float *light, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Object *obj = &objs[tid];
        float lum_d = luminance(obj->mat.diffuse);
        float lum_s = luminance(obj->mat.specular);
        float shine = obj->mat.shininess;
        float lx = light[0], ly = light[1], lz = light[2];
        out[tid] = (lum_d + lum_s * shine) * obj->scale * (lx + ly + lz);
    }
}

__global__ void copy_material(Material *dst, Material *src, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        dst[tid].diffuse.r = src[tid].diffuse.r;
        dst[tid].diffuse.g = src[tid].diffuse.g;
        dst[tid].diffuse.b = src[tid].diffuse.b;
        dst[tid].diffuse.a = src[tid].diffuse.a;
        dst[tid].shininess  = src[tid].shininess;
    }
}
