// Probe: stress test of the inline-merge liveness fix — multiple sequential
//        ternary-result inlines, do-while as struct carrier, switch in loop

struct MinMax { float mn; float mx; };

__device__ MinMax make_minmax(float a, float b) {
    MinMax r;
    r.mn = (a < b) ? a : b;
    r.mx = (a > b) ? a : b;
    return r;
}

__device__ MinMax merge_minmax(MinMax x, MinMax y) {
    MinMax r;
    r.mn = (x.mn < y.mn) ? x.mn : y.mn;
    r.mx = (x.mx > y.mx) ? x.mx : y.mx;
    return r;
}

// Three sequential ternary-heavy inlines per iteration: each returns MinMax.
// Tests that liveness fix scales to multiple inline calls in a chain.
__global__ void triple_minmax(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        MinMax total; total.mn = 1e30f; total.mx = -1e30f;
        for (int i = 0; i < n; i++) {
            MinMax ab = make_minmax(a[i], b[i]);
            MinMax bc = make_minmax(b[i], c[i]);
            MinMax merged = merge_minmax(ab, bc);
            total = merge_minmax(total, merged);
        }
        out[0] = total.mn;
        out[1] = total.mx;
    }
}

// ---------------------------------------------------------------

struct Bucket { float sum; int cnt; int tag; };

// Switch inside an inline — assigns struct fields based on tag
__device__ Bucket classify_bucket(float x, int tag) {
    Bucket b; b.sum = 0.0f; b.cnt = 0; b.tag = tag;
    switch (tag) {
        case 0: b.sum = x;          b.cnt = 1; break;
        case 1: b.sum = x * 2.0f;   b.cnt = 2; break;
        case 2: b.sum = x * x;      b.cnt = 1; break;
        default: b.tag = -1; break;
    }
    return b;
}

__device__ Bucket add_buckets(Bucket a, Bucket b) {
    if (a.tag != b.tag) { a.tag = -1; return a; }
    a.sum += b.sum;
    a.cnt += b.cnt;
    return a;
}

// Switch in inline + struct accumulation + conditional struct merge
__global__ void bucket_accumulate(float *out, float *in, int *tags, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Bucket acc; acc.sum = 0.0f; acc.cnt = 0; acc.tag = 0;
        for (int i = 0; i < n; i++) {
            Bucket b = classify_bucket(in[i], tags[i]);
            acc = add_buckets(acc, b);
        }
        out[0] = acc.sum;
        out[1] = (float)acc.cnt;
        out[2] = (float)acc.tag;
    }
}

// ---------------------------------------------------------------

struct Accum { float val; int iters; };

// do-while as struct carrier: loops until convergence
__global__ void do_while_accum(float *out, float *in, int n, float tol) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Accum a; a.val = 0.0f; a.iters = 0;
        int i = 0;
        do {
            a.val += in[i];
            a.iters++;
            i++;
        } while (i < n && a.val < tol);
        out[0] = a.val;
        out[1] = (float)a.iters;
    }
}
