// Probe: compound assignment on struct fields after inline struct assignment
// - s.val += x after s = fn(s)
// - s.count -= 1 after inline
// - Mixed: s.a = fn1(s).a; s.b = fn2(s).b (field-by-field from different inlines)
// - Pre/postfix on field immediately after inline assignment

struct Acc { float sum, sum2; int count; };
struct Range { float lo, hi; };

__device__ Acc add_val(Acc a, float x) {
    Acc r;
    r.sum   = a.sum + x;
    r.sum2  = a.sum2 + x * x;
    r.count = a.count + 1;
    return r;
}

__device__ Range extend(Range r, float v) {
    Range out;
    out.lo = v < r.lo ? v : r.lo;
    out.hi = v > r.hi ? v : r.hi;
    return out;
}

// Compound += on field after inline
__global__ void compound_after_inline(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc a; a.sum = 0.0f; a.sum2 = 0.0f; a.count = 0;
        for (int i = 0; i < n; i++) {
            a = add_val(a, in[i]);
            a.sum += 10.0f;   // compound += after inline reassignment
            a.count--;        // postfix -- on count after inline
        }
        out[0] = a.sum; out[1] = a.sum2; out[2] = (float)a.count;
    }
}

// Extend range, then check lo/hi with compound assignments
__global__ void extend_and_adjust(float *out, float *in, float margin, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Range r; r.lo = 1e9f; r.hi = -1e9f;
        for (int i = 0; i < n; i++) {
            r = extend(r, in[i]);
        }
        r.lo -= margin;   // compound -= after loop
        r.hi += margin;
        out[0] = r.lo; out[1] = r.hi;
    }
}
