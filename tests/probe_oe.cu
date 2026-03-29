// Probe: nested structs, __device__ global variables, struct return from device fn
// Tests areas not yet covered by the existing test suite.

// ------------------------------------------------------------------
// Nested struct: outer struct has an inner struct field.
// Access via s.inner.x must compute the correct byte offset chain.

struct Inner { float x; float y; };
struct Outer { Inner center; float radius; };

__global__ void nested_struct(float *out, float cx, float cy, float r) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Outer o;
        o.center.x = cx;
        o.center.y = cy;
        o.radius = r;
        out[0] = o.center.x;
        out[1] = o.center.y;
        out[2] = o.radius;
    }
}

// ------------------------------------------------------------------
// __device__ global variable: a single int shared across the grid.
// Must be declared as .global .s32 in PTX module scope.

__device__ int g_counter;

__global__ void global_var_write(int val) {
    int tid = threadIdx.x;
    if (tid == 0) {
        g_counter = val;
    }
}

__global__ void global_var_read(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = g_counter;
    }
}

// ------------------------------------------------------------------
// Struct passed by value to a device function (copy semantics).
// Modifications inside the callee must NOT affect the caller's copy.

struct Vec3 { float x; float y; float z; };

__device__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void struct_by_value(float *out, float ax, float ay, float az,
                                             float bx, float by, float bz) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec3 a; a.x = ax; a.y = ay; a.z = az;
        Vec3 b; b.x = bx; b.y = by; b.z = bz;
        out[0] = dot(a, b);
    }
}
