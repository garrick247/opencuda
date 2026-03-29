// Probe: struct-with-array-field, arrow access, 64-bit shifts, __launch_bounds__
// Tests edge cases flagged as future bug risks in the memory notes.

// ------------------------------------------------------------------
// Struct with scalar fields accessed via pointer arrow syntax.
// ptr->field must work the same as (*ptr).field.

struct Vec2 { float x; float y; };

__device__ void update_vec(Vec2 *p, float dx, float dy) {
    p->x += dx;
    p->y += dy;
}

__global__ __launch_bounds__(256, 4) void arrow_access(float *out, float dx, float dy) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec2 v; v.x = 1.0f; v.y = 2.0f;
        update_vec(&v, dx, dy);
        out[0] = v.x;
        out[1] = v.y;
    }
}

// ------------------------------------------------------------------
// 64-bit shift: shifts on long long values must use b64 instructions.
// The shift amount must remain b32 (not widened to b64).

__global__ void shift64(unsigned long long *out, unsigned long long *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = data[tid];
        out[tid*2+0] = v << 3;    // shl.b64: 64-bit value, 32-bit shift count
        out[tid*2+1] = v >> 3;    // shr.b64: same
    }
}

// ------------------------------------------------------------------
// Const + restrict pointer: must emit ld.global.nc (non-caching load).

__global__ void nc_load(float *out, const float * __restrict__ in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Ternary with struct fields: each branch returns a different field value.
// Tests that ternary type inference correctly handles struct field access.

struct Pair2 { float lo; float hi; };

__global__ void ternary_field(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        Pair2 p; p.lo = -1.0f; p.hi = 1.0f;
        float result = (v > 0.0f) ? p.hi : p.lo;
        out[tid] = result;
    }
}
