// Regression: struct aggregate initializer — Vec3 v = {1.0f, 2.0f, 3.0f};
// Without fix: _parse_assign_expr called on '{' → ParseError "unexpected token '{'".
// Fix: struct declaration's = branch checks for LBRACE; if found, parses
//   { expr, expr, ... } and assigns each value to the corresponding scalar field.

typedef struct { float x, y, z; } Vec3;
typedef struct { int r, g, b, a; } Color;
typedef struct { float u, v; } UV;

__global__ void struct_init_const(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 up = {0.0f, 1.0f, 0.0f};
        Vec3 right = {1.0f, 0.0f, 0.0f};

        // Dot product of up and right (should be 0)
        float dot = up.x * right.x + up.y * right.y + up.z * right.z;
        out[tid] = dot;
    }
}

__global__ void struct_init_dynamic(float *positions, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float px = positions[tid * 3 + 0];
        float py = positions[tid * 3 + 1];
        float pz = positions[tid * 3 + 2];
        Vec3 pos = {px, py, pz};
        out[tid] = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    }
}

__global__ void color_init(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Color red = {255, 0, 0, 255};
        Color green = {0, 255, 0, 255};
        out[tid] = red.r + green.g;
    }
}
