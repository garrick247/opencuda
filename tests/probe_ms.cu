// Probe: Struct patterns in shared memory and complex struct layouts
// - Struct stored in __shared__ array
// - Nested struct with more than 2 levels
// - Struct with mixed int/float fields
// - Struct assignment chain (a = b = c)
// - Struct passed by value to multiple inlined functions
// - Struct comparison via field-by-field

struct RGB {
    unsigned char r, g, b, pad;
};

struct Pixel {
    float r, g, b, a;
};

struct Material {
    Pixel diffuse;
    Pixel specular;
    float shininess;
};

// __shared__ struct array
__global__ void shared_struct_sort(Pixel *out, Pixel *in, int n) {
    __shared__ Pixel tile[32];
    int tid = threadIdx.x;
    if (tid < n) {
        tile[tid] = in[tid];
    }
    __syncthreads();
    if (tid < n) {
        out[tid] = tile[(tid + 1) % n];
    }
}

// Deeply nested struct field access
__global__ void nested_material(float *out, Material *mats, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Material m = mats[tid];
        float d = m.diffuse.r + m.diffuse.g + m.diffuse.b;
        float s = m.specular.r + m.specular.g + m.specular.b;
        out[tid] = d * m.shininess + s;
    }
}

// Mixed int/float struct
struct Transform2D {
    float tx, ty;
    float scale;
    int flags;
};

__device__ Transform2D combine(Transform2D a, Transform2D b) {
    Transform2D r;
    r.tx = a.tx + b.tx * a.scale;
    r.ty = a.ty + b.ty * a.scale;
    r.scale = a.scale * b.scale;
    r.flags = a.flags | b.flags;
    return r;
}

__global__ void transform_chain(float *out, Transform2D *ts, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Transform2D acc = ts[0];
        for (int i = 1; i < n; i++) {
            acc = combine(acc, ts[i]);
        }
        out[0] = acc.tx;
        out[1] = acc.ty;
        out[2] = acc.scale;
        out[3] = (float)acc.flags;
    }
}

// Struct with pointer passed to multiple inlined fns
__device__ float pixel_luma(Pixel p) {
    return 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
}

__device__ Pixel pixel_scale(Pixel p, float s) {
    Pixel out;
    out.r = p.r * s;
    out.g = p.g * s;
    out.b = p.b * s;
    out.a = p.a;
    return out;
}

__global__ void pixel_process(Pixel *out, Pixel *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pixel p = in[tid];
        float luma = pixel_luma(p);
        Pixel scaled = pixel_scale(p, luma > 0.5f ? 0.5f : 1.5f);
        out[tid] = scaled;
    }
}
