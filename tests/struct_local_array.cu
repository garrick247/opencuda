// Regression: local array of structs: Foo arr[N];
// Without fix: ParseError "expected SEMI, got LBRACKET '['" — the [N] after struct
//   name was not consumed (struct declarator returned before array-size handling)
// Also: PTX .align was emitting non-power-of-2 alignment for structs with odd sizes
// Fix 1: struct declarator checks for LBRACKET before creating sentinel; if found,
//         allocates arr as .local PtrTy(StructTy, LOCAL) of N elements.
// Fix 2: emit.py rounds .local .align up to next power of 2.

struct Vec3f {
    float x, y, z;   // size=12, next pow2 align=16
};

struct Coef {
    float a, b;
    int n;
};

__global__ void struct_arr_local(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Local array of 4 Vec3f structs
        Vec3f pts[4];
        pts[0].x = 1.0f; pts[0].y = 0.0f; pts[0].z = 0.0f;
        pts[1].x = 0.0f; pts[1].y = 1.0f; pts[1].z = 0.0f;
        pts[2].x = 0.0f; pts[2].y = 0.0f; pts[2].z = 1.0f;
        pts[3].x = 0.5f; pts[3].y = 0.5f; pts[3].z = 0.5f;
        int i = tid % 4;
        out[tid] = pts[i].x + pts[i].y + pts[i].z;
    }
}

__global__ void coef_arr_local(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Coef coefs[2];
        coefs[0].a = 0.5f; coefs[0].b = 0.5f; coefs[0].n = 2;
        coefs[1].a = 0.25f; coefs[1].b = 0.75f; coefs[1].n = 4;
        int i = tid % 2;
        out[tid] = coefs[i].a * in[tid] + coefs[i].b;
    }
}
