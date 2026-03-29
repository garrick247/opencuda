// Probe: successive inlines to same fn, struct field as loop condition,
//        direct field access on inline result (no named variable)

struct Acc { float sum; int count; };
struct Vec2 { float x, y; };

__device__ Acc step(Acc a, float x) {
    Acc r;
    r.sum   = a.sum + x;
    r.count = a.count + 1;
    return r;
}

__device__ Vec2 make_v2(float x, float y) {
    Vec2 r; r.x = x; r.y = y; return r;
}

// Two successive calls to same inline in one loop body
// Expected: count = 2*n, sum = sum(a[i] + b[i]) for i in 0..n-1
__global__ void double_step(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc s;
        s.sum   = 0.0f;
        s.count = 0;
        for (int i = 0; i < n; i++) {
            s = step(s, a[i]);    // first call
            s = step(s, b[i]);    // second call same iteration
        }
        out[0] = s.sum;
        out[1] = (float)s.count;
    }
}

// Struct field as while-loop condition
// Expected: sum of in[0..k-1] where k = first index where a.count reaches n
__global__ void while_on_field(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc a;
        a.sum   = 0.0f;
        a.count = 0;
        int i = 0;
        while (a.count < n) {
            a = step(a, in[i]);
            i++;
        }
        out[0] = a.sum;
        out[1] = (float)a.count;
    }
}

// Direct field access on inline result (no named variable)
// Expected: sum of make_v2(in[i], in[i]+1.0).x for each i
__global__ void direct_field_on_result(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sx = 0.0f;
        float sy = 0.0f;
        for (int i = 0; i < n; i++) {
            sx += make_v2(in[i], in[i] + 1.0f).x;
            sy += make_v2(in[i], in[i] + 1.0f).y;
        }
        out[0] = sx;
        out[1] = sy;
    }
}
