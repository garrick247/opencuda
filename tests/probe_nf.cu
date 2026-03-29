// Probe: struct with unsigned int field, switch/case in inline,
//        two different inlines called per loop iteration

struct UStats { float sum; unsigned int count; unsigned int flags; };

// classify: returns 0/1/2/3 tag based on value quadrant
__device__ unsigned int classify_val(float x, float threshold) {
    if (x < 0.0f)       return 0u;
    if (x < threshold)  return 1u;
    if (x < threshold * 2.0f) return 2u;
    return 3u;
}

// update: switch on tag, accumulate differently per bucket
__device__ UStats update_ustats(UStats s, float x, unsigned int tag) {
    switch (tag) {
        case 0: s.sum -= x;         break;
        case 1: s.sum += x;         s.count += 1u; break;
        case 2: s.sum += x * 2.0f;  s.count += 2u; break;
        default: s.flags += 1u;     break;
    }
    return s;
}

// Kernel 1: classify + update per element, struct has unsigned fields
__global__ void classify_and_update(float *out, float *in, int n, float threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        UStats s; s.sum = 0.0f; s.count = 0u; s.flags = 0u;
        for (int i = 0; i < n; i++) {
            unsigned int tag = classify_val(in[i], threshold);
            s = update_ustats(s, in[i], tag);
        }
        out[0] = s.sum;
        out[1] = (float)s.count;
        out[2] = (float)s.flags;
    }
}

// ---------------------------------------------------------------

struct Pair { float lo; float hi; };

__device__ Pair make_pair(float a, float b) {
    Pair p;
    p.lo = (a < b) ? a : b;
    p.hi = (a > b) ? a : b;
    return p;
}

__device__ Pair widen_pair(Pair p, float margin) {
    p.lo -= margin;
    p.hi += margin;
    return p;
}

// Kernel 2: two different inlines per iteration, result struct accumulated
__global__ void pair_accumulate(float *out, float *a, float *b, int n, float margin) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Pair total; total.lo = 1e30f; total.hi = -1e30f;
        for (int i = 0; i < n; i++) {
            Pair p = make_pair(a[i], b[i]);
            p = widen_pair(p, margin);
            if (p.lo < total.lo) total.lo = p.lo;
            if (p.hi > total.hi) total.hi = p.hi;
        }
        out[0] = total.lo;
        out[1] = total.hi;
    }
}

// ---------------------------------------------------------------

struct RGB { float r; float g; float b; };

__device__ RGB lerp_rgb(RGB a, RGB b, float t) {
    RGB c;
    c.r = a.r + (b.r - a.r) * t;
    c.g = a.g + (b.g - a.g) * t;
    c.b = a.b + (b.b - a.b) * t;
    return c;
}

__device__ float luma(RGB c) {
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

// Kernel 3: struct returned from inline used as arg to another inline
// lerp_rgb produces RGB; luma consumes it → result is a scalar
__global__ void lerp_and_luma(float *out, float *r0, float *g0, float *b0,
                               float *r1, float *g1, float *b1, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum_luma = 0.0f;
        for (int i = 0; i < n; i++) {
            RGB a; a.r = r0[i]; a.g = g0[i]; a.b = b0[i];
            RGB bv; bv.r = r1[i]; bv.g = g1[i]; bv.b = b1[i];
            RGB mid = lerp_rgb(a, bv, 0.5f);
            sum_luma += luma(mid);
        }
        out[0] = sum_luma;
    }
}
