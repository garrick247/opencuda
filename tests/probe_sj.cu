// Probe: enum types, typedef chains, complex macro expansion,
// and C preprocessor edge cases.

// Enum used as array index and comparison value.
enum Color { RED = 0, GREEN = 1, BLUE = 2, ALPHA = 3 };
enum Status { OK = 0, ERR_BOUNDS = -1, ERR_NAN = -2, ERR_OVERFLOW = 100 };

// Typedef chain.
typedef int MyInt;
typedef MyInt Index;
typedef float MyFloat;

// Macro with expression.
#define SQ(x)      ((x) * (x))
#define CLAMP(v,lo,hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))
#define MAX3(a,b,c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

// ------------------------------------------------------------------
// Enum as array index.

__global__ void enum_index(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int c = tid % 4;
        float r = 0.0f;
        switch (c) {
            case RED:   r = in[tid] * 1.0f; break;
            case GREEN: r = in[tid] * 2.0f; break;
            case BLUE:  r = in[tid] * 3.0f; break;
            case ALPHA: r = in[tid] * 4.0f; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Enum comparison.

__device__ int validate(float v) {
    if (v < -1e6f || v > 1e6f) return ERR_OVERFLOW;
    if (v != v) return ERR_NAN;  // NaN check: v != v is true for NaN
    return OK;
}

__global__ void enum_status(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = validate(in[tid]);
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Typedef usage.

__global__ void typedef_kernel(MyFloat *out, MyFloat *in, Index n) {
    Index tid = (Index)threadIdx.x;
    if (tid < n) {
        MyFloat v = in[tid];
        out[tid] = SQ(v) + CLAMP(v, -1.0f, 1.0f);
    }
}

// ------------------------------------------------------------------
// Macro with side-effect-free multiple evaluation.

__global__ void macro_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sq = SQ(v);
        int cl = CLAMP(v, 0, 255);
        int m3 = MAX3(v, v + 1, v - 1);
        out[tid] = sq + cl + m3;
    }
}

// ------------------------------------------------------------------
// Complex enum switch with fallthrough and default.

__global__ void enum_switch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int code = in[tid] % 6;
        int r;
        switch (code) {
            case 0:
            case 1: r = 10; break;
            case 2: r = 20; break;
            case 3:
            case 4: r = 30; break;
            default: r = 99; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Typedef struct.

typedef struct { float x, y, z; } Vec3f;

__device__ float dot3(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void vec3_dot(float *out, Vec3f *a, Vec3f *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dot3(a[tid], b[tid]);
    }
}
