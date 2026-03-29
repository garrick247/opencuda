// Probe: struct initializer lists, zero-initializers, aggregate
// assignment, and struct literals in expressions.

struct Vec2 {
    float x, y;
};

struct Vec3 {
    float x, y, z;
};

struct Rect {
    int x0, y0, x1, y1;
};

// ------------------------------------------------------------------
// Brace-initializer for struct.

__global__ void struct_brace_init(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v = {in[tid * 3], in[tid * 3 + 1], in[tid * 3 + 2]};
        float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
        out[tid] = len2;
    }
}

// ------------------------------------------------------------------
// Zero-initializer for struct.

__global__ void struct_zero_init(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Rect r = {0, 0, 0, 0};
        r.x1 = tid;
        r.y1 = tid * 2;
        out[tid] = (r.x1 - r.x0) * (r.y1 - r.y0);
    }
}

// ------------------------------------------------------------------
// Struct with partial initializer (remaining fields zero).

__global__ void struct_partial_init(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v = {(float)tid, 0.0f};  // z defaults to 0
        out[tid] = v.x + v.y + v.z;
    }
}

// ------------------------------------------------------------------
// Return struct from device function, use in caller.

__device__ Vec2 make_vec2(float x, float y) {
    Vec2 r;
    r.x = x;
    r.y = y;
    return r;
}

__device__ float vec2_dot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

__global__ void struct_return_use(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 a = make_vec2(in[tid * 2], in[tid * 2 + 1]);
        Vec2 b = make_vec2(in[tid * 2 + 1], in[tid * 2]);
        out[tid] = vec2_dot(a, b);
    }
}

// ------------------------------------------------------------------
// Array of struct on stack with loop initialization.

__global__ void struct_array_stack(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 pts[4];
        for (int i = 0; i < 4; i++) {
            pts[i].x = in[tid * 8 + i * 2];
            pts[i].y = in[tid * 8 + i * 2 + 1];
        }
        // Compute centroid
        float cx = 0.0f, cy = 0.0f;
        for (int i = 0; i < 4; i++) {
            cx += pts[i].x;
            cy += pts[i].y;
        }
        out[tid * 2]     = cx * 0.25f;
        out[tid * 2 + 1] = cy * 0.25f;
    }
}

// ------------------------------------------------------------------
// Struct copy (assignment between struct variables).

__global__ void struct_copy(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 src;
        src.x = in[tid * 3];
        src.y = in[tid * 3 + 1];
        src.z = in[tid * 3 + 2];
        Vec3 dst = src;  // struct copy
        dst.x *= 2.0f;
        out[tid] = dst.x + dst.y + dst.z;
    }
}
