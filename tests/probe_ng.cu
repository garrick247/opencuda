// Probe: struct field as loop step, nested ternary into struct field,
//        early return from inside inline's internal loop

struct Range { float lo; float hi; float step; };
struct Result { float val; int found; };

// Returns first value in range where predicate holds (val > target)
// Has return inside the for-loop body — early exit from inline-internal loop
__device__ Result find_first(Range r, float target) {
    Result res; res.val = 0.0f; res.found = 0;
    for (float v = r.lo; v <= r.hi; v += r.step) {
        if (v > target) {
            res.val = v;
            res.found = 1;
            return res;
        }
    }
    return res;
}

// Kernel 1: struct field as float loop step, early-return inline
__global__ void scan_range(float *out, float lo, float hi, float step, float target) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Range r; r.lo = lo; r.hi = hi; r.step = step;
        Result res = find_first(r, target);
        out[0] = res.val;
        out[1] = (float)res.found;
    }
}

// ---------------------------------------------------------------

struct Bounds { float mn; float mx; int cnt; };

// nested ternary in struct field assignment
__device__ Bounds update_bounds(Bounds b, float x) {
    b.mn = (x < b.mn) ? x : b.mn;
    b.mx = (x > b.mx) ? x : b.mx;
    b.cnt = b.cnt + 1;
    return b;
}

// Kernel 2: accumulate bounds with nested ternary fields per iteration
__global__ void running_bounds(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Bounds b;
        b.mn = 1e30f;
        b.mx = -1e30f;
        b.cnt = 0;
        for (int i = 0; i < n; i++) {
            b = update_bounds(b, in[i]);
        }
        out[0] = b.mn;
        out[1] = b.mx;
        out[2] = (float)b.cnt;
    }
}

// ---------------------------------------------------------------

struct Poly { float a2; float a1; float a0; };  // a2*x^2 + a1*x + a0

__device__ float eval_poly(Poly p, float x) {
    return p.a2 * x * x + p.a1 * x + p.a0;
}

// Two polynomials evaluated per iteration, difference accumulated
__global__ void poly_diff_sum(float *out, float *x_arr, int n,
                               float a2, float a1, float a0,
                               float b2, float b1, float b0) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Poly pa; pa.a2 = a2; pa.a1 = a1; pa.a0 = a0;
        Poly pb; pb.a2 = b2; pb.a1 = b1; pb.a0 = b0;
        float diff_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float x = x_arr[i];
            float va = eval_poly(pa, x);
            float vb = eval_poly(pb, x);
            diff_sum += va - vb;
        }
        out[0] = diff_sum;
    }
}
