// Probe: type synonym coverage (long int, short int, signed int, char),
// complex while conditions with side effects, nested struct member chains,
// multi-argument device functions, and heavy macro expansion.

#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define SQ(x)    ((x)*(x))
#define SWAP(a,b,T) do { T _t = (a); (a) = (b); (b) = _t; } while(0)
#define ABS(x)   ((x) < 0 ? -(x) : (x))

// ------------------------------------------------------------------
// Type synonyms: long int, short int, signed int, char.

__global__ void type_synonyms(long int *out_li, short int *out_si,
                                signed int *out_i,
                                char *out_c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long int  li = (long int)tid * 100L;
        short int si = (short int)(tid & 0xFFFF);
        signed int i = -tid;
        char c = (char)(tid & 127);
        out_li[tid] = li;
        out_si[tid] = si;
        out_i[tid]  = i;
        out_c[tid]  = c;
    }
}

// ------------------------------------------------------------------
// Macro-heavy kernel: CLAMP, SQ, ABS in tight loops.

__global__ void macro_kernel(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float s = SQ(v);
        float a = ABS(v - 5.0f);
        float c = CLAMP(s - a, 0.0f, 100.0f);
        out[tid] = c;
    }
}

// ------------------------------------------------------------------
// Nested struct with array member.

struct Vec2i { int x, y; };

struct AABB {
    struct Vec2i lo;
    struct Vec2i hi;
    int tag;
};

__device__ int aabb_area(struct AABB box) {
    int w = box.hi.x - box.lo.x;
    int h = box.hi.y - box.lo.y;
    return w * h;
}

__device__ int aabb_contains(struct AABB box, struct Vec2i p) {
    return p.x >= box.lo.x && p.x <= box.hi.x &&
           p.y >= box.lo.y && p.y <= box.hi.y;
}

__global__ void aabb_kernel(int *out_area, int *out_hit,
                              int *blo_x, int *blo_y,
                              int *bhi_x, int *bhi_y,
                              int *px, int *py, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct AABB box;
        box.lo.x = blo_x[tid]; box.lo.y = blo_y[tid];
        box.hi.x = bhi_x[tid]; box.hi.y = bhi_y[tid];
        box.tag = tid;
        struct Vec2i p;
        p.x = px[tid]; p.y = py[tid];
        out_area[tid] = aabb_area(box);
        out_hit[tid]  = aabb_contains(box, p);
    }
}

// ------------------------------------------------------------------
// Complex while condition with side effects (pre-increment in cond).

__global__ void while_sideeffect(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0, i = 0;
        // Condition has side effect: ++i
        while (++i <= 8 && v > 0) {
            sum += v & 1;
            v >>= 1;
        }
        out[tid] = sum;  // popcount of lower 8 bits
    }
}

// ------------------------------------------------------------------
// SWAP macro test (uses do-while(0) trick).

__global__ void swap_kernel(int *out_a, int *out_b, int *in_a, int *in_b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in_a[tid];
        int b = in_b[tid];
        if (a > b) {
            SWAP(a, b, int);
        }
        out_a[tid] = a;  // min
        out_b[tid] = b;  // max
    }
}

// ------------------------------------------------------------------
// Device function with 10 parameters.

__device__ float poly10(float x, float a0, float a1, float a2, float a3,
                          float a4, float a5, float a6, float a7, float a8) {
    // Horner's method: a0 + x*(a1 + x*(a2 + ... + x*a8))
    float r = a8;
    r = a7 + x * r;
    r = a6 + x * r;
    r = a5 + x * r;
    r = a4 + x * r;
    r = a3 + x * r;
    r = a2 + x * r;
    r = a1 + x * r;
    r = a0 + x * r;
    return r;
}

__global__ void poly10_kernel(float *out, float *xs, float *coeffs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = xs[tid];
        out[tid] = poly10(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                          coeffs[4], coeffs[5], coeffs[6], coeffs[7], coeffs[8]);
    }
}

// ------------------------------------------------------------------
// Unsigned char (byte) manipulation.

__global__ void uchar_ops(unsigned char *out, unsigned char *a,
                            unsigned char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int av = (unsigned int)a[tid];
        unsigned int bv = (unsigned int)b[tid];
        unsigned int xor_v = av ^ bv;
        unsigned int and_v = av & bv;
        out[tid] = (unsigned char)((xor_v + and_v) & 0xFF);
    }
}
