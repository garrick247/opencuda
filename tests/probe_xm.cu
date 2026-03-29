// Probe: nested struct operations, multi-kernel shared device fn,
// long basic block CSE, and function with many parameters.

// ------------------------------------------------------------------
// Nested struct: Point inside Rect.

struct Point { float x, y; };
struct Rect  { struct Point lo, hi; };

__device__ float rect_area(struct Rect r) {
    float w = r.hi.x - r.lo.x;
    float h = r.hi.y - r.lo.y;
    return w * h;
}

__device__ int point_in_rect(struct Point p, struct Rect r) {
    return p.x >= r.lo.x && p.x <= r.hi.x &&
           p.y >= r.lo.y && p.y <= r.hi.y;
}

__global__ void rect_ops(float *out_area, int *out_hit,
                          float *px, float *py,
                          float *rlo_x, float *rlo_y,
                          float *rhi_x, float *rhi_y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Rect r;
        r.lo.x = rlo_x[tid]; r.lo.y = rlo_y[tid];
        r.hi.x = rhi_x[tid]; r.hi.y = rhi_y[tid];
        struct Point p;
        p.x = px[tid]; p.y = py[tid];
        out_area[tid] = rect_area(r);
        out_hit[tid]  = point_in_rect(p, r);
    }
}

// ------------------------------------------------------------------
// Same device fn called from two kernels: ensure inlining is consistent.

__device__ float sigmoid(float x) {
    float ex = expf(-x);
    return 1.0f / (1.0f + ex);
}

__global__ void sigmoid_forward(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = sigmoid(in[tid]);
}

__global__ void sigmoid_derivative(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float s = sigmoid(in[tid]);
        out[tid] = s * (1.0f - s);
    }
}

// ------------------------------------------------------------------
// Device fn with 8 parameters.

__device__ float blend8(float a, float b, float c, float d,
                          float e, float f, float g, float h,
                          float w) {
    return w * (a + b + c + d) + (1.0f - w) * (e + f + g + h);
}

__global__ void blend_kernel(float *out, float *a, float *b, float *c, float *d,
                               float *e, float *f, float *g, float *h,
                               float *w, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = blend8(a[tid], b[tid], c[tid], d[tid],
                          e[tid], f[tid], g[tid], h[tid], w[tid]);
    }
}

// ------------------------------------------------------------------
// Long basic block: 30 sequential operations on same values (CSE target).

__global__ void long_bb_cse(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int s = 0;
        // Each pair of lines computes same thing: CSE should fire for repeated subexprs
        int a = v * 3 + 1;
        int b = v * 3 + 1;    // same as a — CSE should share
        s += a + b;

        int c = (v >> 2) & 0xFF;
        int d = (v >> 2) & 0xFF;  // same as c
        s += c + d;

        int e = v * v;
        int f = v * v;        // same as e
        s += e + f;

        int g = (a + c) * 2;
        int h = (a + c) * 2;  // same as g
        s += g + h;

        out[tid] = s;
        // If CSE works: s = 2*(v*3+1) + 2*((v>>2)&0xFF) + 2*(v*v) + 2*((v*3+1+(v>>2)&0xFF)*2)
    }
}

// ------------------------------------------------------------------
// Struct with 3 levels of nesting.

struct V3 { float x, y, z; };
struct Transform3D { struct V3 pos; struct V3 scale; };

__device__ struct V3 apply_transform3d(struct V3 v, struct Transform3D t) {
    struct V3 r;
    r.x = v.x * t.scale.x + t.pos.x;
    r.y = v.y * t.scale.y + t.pos.y;
    r.z = v.z * t.scale.z + t.pos.z;
    return r;
}

__global__ void transform3d_kernel(float *ox, float *oy, float *oz,
                                    float *ix, float *iy, float *iz,
                                    float px, float py, float pz,
                                    float sx, float sy, float sz, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        struct V3 v;
        v.x = ix[tid]; v.y = iy[tid]; v.z = iz[tid];
        struct Transform3D t;
        t.pos.x = px; t.pos.y = py; t.pos.z = pz;
        t.scale.x = sx; t.scale.y = sy; t.scale.z = sz;
        struct V3 r = apply_transform3d(v, t);
        ox[tid] = r.x; oy[tid] = r.y; oz[tid] = r.z;
    }
}

// ------------------------------------------------------------------
// Multiple device functions with same signature, different logic.

__device__ float relu(float x)         { return x > 0.0f ? x : 0.0f; }
__device__ float leaky_relu(float x)   { return x > 0.0f ? x : 0.01f * x; }
__device__ float tanh_approx(float x) {
    // Pade approximation: tanh(x) ≈ x*(27+x^2) / (27+9*x^2)
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

__global__ void activations(float *r, float *lr, float *ta, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        r[tid]  = relu(v);
        lr[tid] = leaky_relu(v);
        ta[tid] = tanh_approx(v);
    }
}
