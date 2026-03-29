// Probe: Struct-to-struct copy semantics
// - Direct struct copy: s2 = s1, then mutate s2 fields — s1 unchanged
// - Struct assigned in one branch only (if without else), used after merge
// - Struct returned from device fn, assigned to local, fields accessed in loop
// - Struct copy inside loop body (not via inline return)

struct Vec2 { float x, y; };
struct AABB { float minx, miny, maxx, maxy; };

// Direct copy: result = input, then scale fields
__device__ Vec2 scale2(Vec2 v, float s) {
    Vec2 r = v;
    r.x = v.x * s;
    r.y = v.y * s;
    return r;
}

__global__ void scale_vectors(float *out, float *in, float s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 v;
        v.x = in[tid*2];
        v.y = in[tid*2+1];
        Vec2 r = scale2(v, s);
        out[tid*2]   = r.x;
        out[tid*2+1] = r.y;
    }
}

// Struct assigned in if-branch only, unmodified in else path
__device__ AABB merge_aabb(AABB a, AABB b, int use_b) {
    AABB result = a;
    if (use_b) {
        result.minx = a.minx < b.minx ? a.minx : b.minx;
        result.miny = a.miny < b.miny ? a.miny : b.miny;
        result.maxx = a.maxx > b.maxx ? a.maxx : b.maxx;
        result.maxy = a.maxy > b.maxy ? a.maxy : b.maxy;
    }
    return result;
}

__global__ void merge_boxes(float *out, float *ina, float *inb, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        AABB a; a.minx=ina[tid*4]; a.miny=ina[tid*4+1]; a.maxx=ina[tid*4+2]; a.maxy=ina[tid*4+3];
        AABB b; b.minx=inb[tid*4]; b.miny=inb[tid*4+1]; b.maxx=inb[tid*4+2]; b.maxy=inb[tid*4+3];
        AABB r = merge_aabb(a, b, flags[tid]);
        out[tid*4]=r.minx; out[tid*4+1]=r.miny; out[tid*4+2]=r.maxx; out[tid*4+3]=r.maxy;
    }
}

// Struct copy inside for loop (copy then mutate, no inline return)
__global__ void running_max(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        AABB best;
        best.minx = in[0]; best.miny = in[1]; best.maxx = in[2]; best.maxy = in[3];
        for (int i = 1; i < n; i++) {
            AABB cur;
            cur.minx = in[i*4]; cur.miny = in[i*4+1]; cur.maxx = in[i*4+2]; cur.maxy = in[i*4+3];
            AABB prev = best;
            best.minx = cur.minx < prev.minx ? cur.minx : prev.minx;
            best.miny = cur.miny < prev.miny ? cur.miny : prev.miny;
            best.maxx = cur.maxx > prev.maxx ? cur.maxx : prev.maxx;
            best.maxy = cur.maxy > prev.maxy ? cur.maxy : prev.maxy;
        }
        out[0]=best.minx; out[1]=best.miny; out[2]=best.maxx; out[3]=best.maxy;
    }
}

// Nested struct copy: inner copy assigned to outer accumulator
__device__ Vec2 vec2_add(Vec2 a, Vec2 b) {
    Vec2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

__device__ Vec2 vec2_scale(Vec2 v, float s) {
    Vec2 r = v;
    r.x *= s;
    r.y *= s;
    return r;
}

__global__ void weighted_sum(float *out, float *in, float *weights, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec2 acc; acc.x = 0.0f; acc.y = 0.0f;
        for (int i = 0; i < n; i++) {
            Vec2 v; v.x = in[i*2]; v.y = in[i*2+1];
            Vec2 scaled = vec2_scale(v, weights[i]);
            acc = vec2_add(acc, scaled);
        }
        out[0] = acc.x; out[1] = acc.y;
    }
}
