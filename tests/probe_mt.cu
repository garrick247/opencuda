// Probe: Struct-returning __device__ functions with multiple return paths
// - Early return inside if-block (struct)
// - All branches return struct
// - Struct early return inside nested if
// - While loop accumulating struct via multi-return function
// - Do-while loop accumulating struct via multi-return function

struct Vec3 { float x, y, z; };
struct Interval { float lo, hi; };

// Early return inside if
__device__ Vec3 safe_normalize(Vec3 v) {
    float len = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len < 1e-6f) {
        Vec3 z; z.x=0.0f; z.y=0.0f; z.z=0.0f;
        return z;
    }
    float inv = 1.0f / len;
    Vec3 r;
    r.x = v.x * inv;
    r.y = v.y * inv;
    r.z = v.z * inv;
    return r;
}

__global__ void normalize_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v;
        v.x = in[tid*3];
        v.y = in[tid*3+1];
        v.z = in[tid*3+2];
        Vec3 norm = safe_normalize(v);
        out[tid*3]   = norm.x;
        out[tid*3+1] = norm.y;
        out[tid*3+2] = norm.z;
    }
}

// All branches return struct
__device__ Vec3 classify_vec(float x) {
    if (x > 0.0f) {
        Vec3 r; r.x = x; r.y = 0.0f; r.z = 0.0f;
        return r;
    } else if (x < 0.0f) {
        Vec3 r; r.x = 0.0f; r.y = -x; r.z = 0.0f;
        return r;
    } else {
        Vec3 r; r.x = 0.0f; r.y = 0.0f; r.z = 1.0f;
        return r;
    }
}

__global__ void classify_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v = classify_vec(in[tid]);
        out[tid*3]   = v.x;
        out[tid*3+1] = v.y;
        out[tid*3+2] = v.z;
    }
}

// Interval clamp with early return
__device__ Interval clamp_interval(Interval a, float lo, float hi) {
    if (a.lo >= hi) {
        Interval r; r.lo = hi; r.hi = hi;
        return r;
    }
    if (a.hi <= lo) {
        Interval r; r.lo = lo; r.hi = lo;
        return r;
    }
    Interval r;
    r.lo = a.lo < lo ? lo : a.lo;
    r.hi = a.hi > hi ? hi : a.hi;
    return r;
}

__global__ void clamp_interval_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Interval iv;
        iv.lo = in[tid*2];
        iv.hi = in[tid*2+1];
        Interval clamped = clamp_interval(iv, 0.0f, 1.0f);
        out[tid*2]   = clamped.lo;
        out[tid*2+1] = clamped.hi;
    }
}

// While loop accumulating struct via multi-return function
__device__ Vec3 vec_add(Vec3 a, Vec3 b) {
    Vec3 r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z;
    return r;
}

__global__ void while_struct_multret(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec3 acc; acc.x=0.0f; acc.y=0.0f; acc.z=0.0f;
        int i = 0;
        while (i < n) {
            Vec3 cur;
            cur.x = in[i*3];
            cur.y = in[i*3+1];
            cur.z = in[i*3+2];
            Vec3 norm = safe_normalize(cur);
            acc = vec_add(acc, norm);
            i++;
        }
        out[0] = acc.x; out[1] = acc.y; out[2] = acc.z;
    }
}

// Do-while with struct multi-return in body
__global__ void do_while_struct_multret(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Vec3 acc; acc.x=0.0f; acc.y=0.0f; acc.z=0.0f;
        int i = 0;
        do {
            Vec3 v;
            v.x = in[i*3]; v.y = in[i*3+1]; v.z = in[i*3+2];
            Vec3 norm = safe_normalize(v);
            acc = vec_add(acc, norm);
            i++;
        } while (i < n);
        out[0] = acc.x; out[1] = acc.y; out[2] = acc.z;
    }
}
