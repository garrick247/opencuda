// Probe: Struct initialization and accumulation edge cases
// - Struct initialized via aggregate init {a, b, c}
// - Struct field accessed on result of inline call directly: f(x).field
// - Struct accumulation in while-loop (not for-loop)
// - Two struct types mixed in same kernel
// - Struct copy from one typed accumulator to another

struct Vec3 { float x, y, z; };
struct Stats { float sum, sum2; int count; };

// Aggregate init and direct field access on result
__device__ Vec3 make_vec3(float x, float y, float z) {
    Vec3 r; r.x = x; r.y = y; r.z = z;
    return r;
}

__device__ float dot3(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Stats accumulator initialized via field-by-field assignment then accumulated in while
__device__ Stats accum_stats(Stats s, float x) {
    Stats r;
    r.sum = s.sum + x;
    r.sum2 = s.sum2 + x * x;
    r.count = s.count + 1;
    return r;
}

__global__ void compute_stats(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s;
        s.sum = 0.0f;
        s.sum2 = 0.0f;
        s.count = 0;
        int i = 0;
        while (i < n) {
            s = accum_stats(s, in[i]);
            i++;
        }
        out[0] = s.sum;
        out[1] = s.sum2;
        out[2] = (float)s.count;
    }
}

// Two mixed struct types in same kernel
__global__ void dot_and_stats(float *out, float *vecs, float *vals, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec3 acc_v; acc_v.x = 0.0f; acc_v.y = 0.0f; acc_v.z = 0.0f;
        Stats s; s.sum = 0.0f; s.sum2 = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            Vec3 v; v.x = vecs[i*3]; v.y = vecs[i*3+1]; v.z = vecs[i*3+2];
            float d = dot3(v, acc_v);
            s = accum_stats(s, d);
            acc_v.x += v.x; acc_v.y += v.y; acc_v.z += v.z;
        }
        out[0] = s.sum; out[1] = s.sum2; out[2] = (float)s.count;
        out[3] = acc_v.x; out[4] = acc_v.y; out[5] = acc_v.z;
    }
}

// Struct accumulation via direct copy in loop body
__device__ Vec3 clamp3(Vec3 v, float lo, float hi) {
    Vec3 r;
    r.x = v.x < lo ? lo : (v.x > hi ? hi : v.x);
    r.y = v.y < lo ? lo : (v.y > hi ? hi : v.y);
    r.z = v.z < lo ? lo : (v.z > hi ? hi : v.z);
    return r;
}

__global__ void clamp_and_sum(float *out, float *in, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec3 total; total.x = 0.0f; total.y = 0.0f; total.z = 0.0f;
        for (int i = 0; i < n; i++) {
            Vec3 v; v.x = in[i*3]; v.y = in[i*3+1]; v.z = in[i*3+2];
            Vec3 c = clamp3(v, lo, hi);
            total.x += c.x; total.y += c.y; total.z += c.z;
        }
        out[0] = total.x; out[1] = total.y; out[2] = total.z;
    }
}
