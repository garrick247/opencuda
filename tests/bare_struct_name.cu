// Regression: bare struct name (C++ style) in sizeof, casts, and variable declarations
// Without fix: _parse_type and _parse_stmt declaration dispatch only checked
//   _typedefs, not _struct_types — raised ParseError "expected type, got 'Vec3'"
//   or ParseError "undefined variable 'Vec3'".
// Fix 1: _parse_type falls through to check _struct_types after _typedefs.
// Fix 2: _parse_stmt declaration guard checks `tok.value in self._struct_types`.

struct Vec3 { float x, y, z; };
struct Ray  { float ox, oy, oz, dx, dy, dz; };

// sizeof with bare struct name
__global__ void sizeof_struct(int *out) {
    out[0] = sizeof(Vec3);   // 12 bytes
    out[1] = sizeof(Ray);    // 24 bytes
}

// C++ style variable declaration: Vec3 v; (no 'struct' keyword)
__global__ void cpp_style_decl(float *out, float *x, float *y, float *z, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v = {x[tid], y[tid], z[tid]};
        out[tid] = v.x * v.x + v.y * v.y + v.z * v.z;
    }
}

// Bare struct name in __device__ function parameter
__device__ float ray_len_sq(Ray r) {
    return r.dx * r.dx + r.dy * r.dy + r.dz * r.dz;
}

__global__ void use_ray(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Ray r;
        r.ox = 0.0f; r.oy = 0.0f; r.oz = 0.0f;
        r.dx = data[tid]; r.dy = data[tid]; r.dz = data[tid];
        out[tid] = ray_len_sq(r);
    }
}
