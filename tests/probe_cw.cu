// Probe: Heavy use of __device__ functions as building blocks
// - 5+ level call chain
// - Mutually used functions (not recursive, just both use a third)
// - Device function returning struct

struct Vec2 {
    float x, y;
};

__device__ float vec2_dot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float vec2_len(Vec2 v) {
    return sqrtf(vec2_dot(v, v));
}

__device__ Vec2 vec2_norm(Vec2 v) {
    float len = vec2_len(v);
    Vec2 r;
    r.x = v.x / len;
    r.y = v.y / len;
    return r;
}

__device__ float vec2_angle(Vec2 a, Vec2 b) {
    Vec2 na = vec2_norm(a);
    Vec2 nb = vec2_norm(b);
    float d = vec2_dot(na, nb);
    // Clamp to [-1, 1] for acos
    if (d > 1.0f) d = 1.0f;
    if (d < -1.0f) d = -1.0f;
    return d;  // would be acos(d) but that's an intrinsic we don't need to test
}

__global__ void vector_angles(float *out, Vec2 *a, Vec2 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = vec2_angle(a[tid], b[tid]);
    }
}
