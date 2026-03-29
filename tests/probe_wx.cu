// Probe: float accumulation patterns, struct-in-loop, ptr aliasing in device fns,
// macro with side effects, and multi-condition loop guards.

// ------------------------------------------------------------------
// Float accumulation with FMA pattern.

__global__ void fma_accumulate(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = c[tid];
        // Unrolled 4x FMA accumulation
        int base = tid * 4;
        if (base + 3 < n) {
            sum = __fmaf_rn(a[base+0], b[base+0], sum);
            sum = __fmaf_rn(a[base+1], b[base+1], sum);
            sum = __fmaf_rn(a[base+2], b[base+2], sum);
            sum = __fmaf_rn(a[base+3], b[base+3], sum);
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Struct in loop: update struct fields per iteration.

struct Stats {
    float sum;
    float sum_sq;
    int   count;
};

__device__ void update_stats(struct Stats *s, float v) {
    s->sum    += v;
    s->sum_sq += v * v;
    s->count  += 1;
}

__device__ float stats_mean(struct Stats *s) {
    return (s->count > 0) ? s->sum / (float)s->count : 0.0f;
}

__global__ void running_stats(float *out_mean, float *out_var,
                               float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Stats s;
        s.sum = 0.0f;
        s.sum_sq = 0.0f;
        s.count = 0;

        for (int i = tid; i < n; i += blockDim.x) {
            update_stats(&s, in[i]);
        }

        float mean = stats_mean(&s);
        float var = (s.count > 0) ?
            (s.sum_sq / (float)s.count - mean * mean) : 0.0f;
        out_mean[tid] = mean;
        out_var[tid]  = var;
    }
}

// ------------------------------------------------------------------
// Pointer aliasing guard: restrict on two output arrays.

__global__ void no_alias_sum(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// Multi-condition loop guard: while (a > 0 && b > 0 && c != 0).

__global__ void multi_cond_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = tid + 1;
        int b = 10;
        int c = tid % 3 + 1;
        int count = 0;
        while (a > 0 && b > 0 && c != 0) {
            a--;
            b -= 2;
            c = (c + 1) % 4;
            count++;
            if (count >= 16) break;
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Macro with argument used twice (potential double-eval issue if not const).

#define SQ(x) ((x) * (x))
#define CUBE(x) ((x) * SQ(x))

__global__ void macro_double_eval(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // v is a load — using in macro twice is fine since it's a register
        out[tid] = SQ(v) + CUBE(v);  // v^2 + v^3
    }
}

// ------------------------------------------------------------------
// Early exit with multiple conditions.

__device__ int classify(int v) {
    if (v < 0) return -1;
    if (v == 0) return 0;
    if (v < 10) return 1;
    if (v < 100) return 2;
    return 3;
}

__global__ void classify_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify(in[tid]);
    }
}

// ------------------------------------------------------------------
// Struct assignment and comparison chain.

struct Interval { int lo; int hi; };

__device__ int overlaps(struct Interval a, struct Interval b) {
    return a.lo <= b.hi && a.hi >= b.lo;
}

__global__ void interval_overlap(int *out, int *lo_a, int *hi_a,
                                  int *lo_b, int *hi_b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Interval a, b;
        a.lo = lo_a[tid]; a.hi = hi_a[tid];
        b.lo = lo_b[tid]; b.hi = hi_b[tid];
        out[tid] = overlaps(a, b);
    }
}

// ------------------------------------------------------------------
// Parallel prefix XOR (Gray code).

__global__ void gray_code_prefix(unsigned int *out, unsigned int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        unsigned int v = in[gid];
        // Gray code: G(n) = n ^ (n >> 1)
        out[gid] = v ^ (v >> 1);
    }
}

// ------------------------------------------------------------------
// Bit counting patterns: parity (XOR of all bits).

__global__ void bit_parity(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Parity via __popc: count set bits, take LSB
        out[tid] = __popc(v) & 1;
    }
}

// ------------------------------------------------------------------
// Conditional assignment with complex expression in condition.

__global__ void complex_cond_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Assign based on bitfield condition
        int bit3 = (v >> 3) & 1;
        int bit5 = (v >> 5) & 1;
        int bit7 = (v >> 7) & 1;
        out[tid] = (bit3 && !bit5) ? v | 0x100 :
                   (bit5 && bit7)  ? v & ~0x100 :
                                     v;
    }
}
