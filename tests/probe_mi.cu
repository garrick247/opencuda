// Probe: Multi-level struct function call chains and complex loop patterns
// - Function returning struct used immediately as arg to another function
// - Struct returned from function modified and stored
// - Do-while loop with struct carried value
// - Loop with struct array and early break
// - Struct with __device__ method-like patterns

struct Vec2 {
    float x, y;
};

struct Transform {
    float scale;
    Vec2 offset;
};

__device__ Vec2 vec2_add(Vec2 a, Vec2 b) {
    Vec2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

__device__ Vec2 vec2_scale(Vec2 v, float s) {
    Vec2 r;
    r.x = v.x * s;
    r.y = v.y * s;
    return r;
}

__device__ float vec2_dot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ Vec2 apply_transform(Vec2 v, Transform t) {
    Vec2 scaled = vec2_scale(v, t.scale);
    Vec2 result = vec2_add(scaled, t.offset);
    return result;
}

// Multi-level chain: apply_transform(vec2_add(a, b), t)
__global__ void multi_level_chain(float *out, Vec2 *pts, Transform t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 a = pts[tid];
        Vec2 b = pts[(tid + 1) % n];
        Vec2 sum = vec2_add(a, b);
        Vec2 result = apply_transform(sum, t);
        out[tid] = result.x + result.y;
    }
}

// Struct carried in a while loop
__global__ void struct_while_loop(float *out, Vec2 *in, int n, float threshold) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 pos;
        pos.x = in[tid].x;
        pos.y = in[tid].y;
        int steps = 0;
        while (vec2_dot(pos, pos) < threshold * threshold && steps < 100) {
            pos.x = pos.x * 1.1f + 0.01f;
            pos.y = pos.y * 0.9f + 0.01f;
            steps++;
        }
        out[tid] = (float)steps;
    }
}

// Do-while with struct
__global__ void struct_dowhile(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 v;
        v.x = in[tid];
        v.y = in[tid] * 2.0f;
        int i = 0;
        do {
            v.x += v.y;
            v.y -= 0.1f;
            i++;
        } while (i < 5);
        out[tid] = v.x;
    }
}

// Array-of-structs with early break
__global__ void struct_array_search(int *out, Vec2 *pts, int n, float target_dist) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 origin;
        origin.x = 0.0f;
        origin.y = 0.0f;
        int found = -1;
        for (int i = 0; i < n; i++) {
            float dx = pts[i].x - origin.x;
            float dy = pts[i].y - origin.y;
            float dist = dx * dx + dy * dy;
            if (dist < target_dist * target_dist) {
                found = i;
                break;
            }
        }
        out[tid] = found;
    }
}

// Transform applied to each element with accumulation
__global__ void transform_reduce(float *out, Vec2 *pts, Transform t, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            Vec2 r = apply_transform(pts[i], t);
            total += r.x * r.x + r.y * r.y;
        }
        out[0] = total;
    }
}
