// Probe: inline return used in condition, struct in nested call chain,
//        device fn with local array, inline result fed to another inline arg

struct Stat2 { float mean; float var; };

__device__ Stat2 compute_stat2(float *arr, int n) {
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum  += arr[i];
        sum2 += arr[i] * arr[i];
    }
    float m = sum / (float)n;
    float v = sum2 / (float)n - m * m;
    Stat2 s; s.mean = m; s.var = v;
    return s;
}

// inline return used in conditional: if (stat.mean > 0) ...
__global__ void stat_conditional(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stat2 s = compute_stat2(in, n);
        if (s.mean > 0.0f) {
            out[0] = s.mean;
            out[1] = s.var;
        } else {
            out[0] = 0.0f;
            out[1] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------

struct Vec3 { float x; float y; float z; };

__device__ Vec3 normalize3(Vec3 v) {
    float len = v.x*v.x + v.y*v.y + v.z*v.z;
    // approximate rsqrt via div
    float inv = 1.0f / len;
    Vec3 r; r.x = v.x * inv; r.y = v.y * inv; r.z = v.z * inv;
    return r;
}

__device__ float dot3(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// inline result passed directly as arg to another inline:
// dot3(normalize3(v1), normalize3(v2)) — each normalize returns Vec3
__global__ void normalized_dot(float *out, float *v1, float *v2, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < n; i += 3) {
            Vec3 a; a.x = v1[i]; a.y = v1[i+1]; a.z = v1[i+2];
            Vec3 b; b.x = v2[i]; b.y = v2[i+1]; b.z = v2[i+2];
            Vec3 na = normalize3(a);
            Vec3 nb = normalize3(b);
            acc += dot3(na, nb);
        }
        out[0] = acc;
    }
}

// ---------------------------------------------------------------

struct Window { int start; int end; };

__device__ Window make_window(int center, int hw, int lo_clamp, int hi_clamp) {
    Window w;
    w.start = (center - hw < lo_clamp) ? lo_clamp : center - hw;
    w.end   = (center + hw > hi_clamp) ? hi_clamp : center + hw;
    return w;
}

__device__ float window_sum(float *arr, Window w) {
    float s = 0.0f;
    for (int k = w.start; k <= w.end; k++) {
        s += arr[k];
    }
    return s;
}

// struct with int fields, passed to second inline as arg
__global__ void windowed_sum(float *out, float *in, int n, int hw) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            Window w = make_window(i, hw, 0, n - 1);
            out[i] = window_sum(in, w);
        }
    }
}
