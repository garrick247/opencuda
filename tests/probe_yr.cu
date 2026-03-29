// Probe: unusual but valid C patterns — struct assigned via memcpy-like pattern,
// multiple nested ternary with function calls as operands, for-loop where
// init is a struct field access, while with assignment in condition,
// #undef and redefine, conditional compilation with #if 0, long chains
// of pointer dereferences (*(*ptr)), and self-referential local computation.

// ------------------------------------------------------------------
// #undef and redefine.

#define VAL 10
#undef VAL
#define VAL 20

__global__ void undef_redef(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = VAL;  // should be 20, not 10
}

// ------------------------------------------------------------------
// #if 0 / #if 1 conditional compilation.

#if 0
// This block should be completely ignored
__global__ void dead_kernel(int *out, int n) {
    out[0] = 99999;  // should never be emitted
}
#endif

#if 1
__global__ void live_kernel(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = 42;
}
#endif

// ------------------------------------------------------------------
// While with assignment in condition (common C idiom).

__device__ int next_token(int *buf, int *pos, int len) {
    if (*pos >= len) return -1;
    return buf[(*pos)++];
}

__global__ void while_assign_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[8];
        for (int k = 0; k < 8; k++) buf[k] = in[(tid + k) % n];
        int pos = 0;
        int sum = 0;
        int tok;
        // Can't do `while ((tok = next_token(...)) != -1)` directly — use do-while
        do {
            tok = next_token(buf, &pos, 8);
            if (tok < 0) break;
            sum += tok;
        } while (1);
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Nested ternary with function calls as all three operands.

__device__ int triple_fn(int x) { return x * 3; }
__device__ int double_fn(int x) { return x * 2; }

__global__ void nested_fn_ternary(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid];
        // Nested ternary where each arm calls a function
        int r = (x > 0) ? triple_fn(x) :
                (x < 0) ? double_fn(-x) :
                          double_fn(y);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// For-loop where init part accesses struct field.

struct Range { int lo, hi; };

__device__ int range_sum(struct Range r) {
    int s = 0;
    for (int i = r.lo; i <= r.hi; i++) s += i;
    return s;
}

__global__ void range_sum_kernel(int *out, int *lo, int *hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Range r;
        r.lo = lo[tid];
        r.hi = hi[tid];
        out[tid] = range_sum(r);
    }
}

// ------------------------------------------------------------------
// Double pointer dereference (*(*ptr)).

__global__ void double_deref(int *out, int **ptrs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // ptrs[tid] points to an int
        out[tid] = *ptrs[tid];
    }
}

// ------------------------------------------------------------------
// Complex: mixing all types — int, float, double, long long in one kernel.

__global__ void all_types_mix(int *out_i, float *out_f, double *out_d,
                                long long *out_ll,
                                int *in_i, float *in_f, double *in_d,
                                long long *in_ll, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int    iv = in_i[tid];
        float  fv = in_f[tid];
        double dv = in_d[tid];
        long long llv = in_ll[tid];
        // Cross-type arithmetic
        out_i[tid]  = iv + (int)fv;
        out_f[tid]  = fv + (float)iv;
        out_d[tid]  = dv + (double)iv;
        out_ll[tid] = llv + (long long)iv;
    }
}

// ------------------------------------------------------------------
// Kernel with only global reads (no writes except last).

__global__ void read_heavy(int *out, int *a, int *b, int *c,
                             int *d, int *e, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid];
        int vd = d[tid], ve = e[tid];
        // Long expression with all reads
        out[tid] = (va + vb) * (vc - vd) + ve * va - vb / (vc + 1);
    }
}
