// Probe: function pointer typedef (parse-only), complex type casts,
// do-while with break/continue, multi-dim local arrays as function args,
// __restrict__ on 3+ pointers, and integer width boundary arithmetic.

// ------------------------------------------------------------------
// __restrict__ on 3 pointers simultaneously (all 3 non-aliasing).

__global__ void triple_restrict(float * __restrict__ out,
                                  const float * __restrict__ a,
                                  const float * __restrict__ b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}

// ------------------------------------------------------------------
// Casting integer expression to different widths.

__global__ void cast_chain(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Chain: int → long long → unsigned → int (modular)
        long long ll = (long long)v * (long long)v;
        unsigned u = (unsigned)(ll & 0xFFFFFFFF);
        int back = (int)u;
        out[tid] = (long long)back + ll;
    }
}

// ------------------------------------------------------------------
// Do-while with break and continue.

__global__ void do_while_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        // Count steps of Collatz sequence until reaching 1 or count > 100
        do {
            if (v == 1) break;
            if (v % 2 == 0) { v /= 2; count++; continue; }
            v = 3 * v + 1;
            count++;
        } while (count < 100);
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Nested loop with labeled-continue semantics via flag.

__global__ void nested_loop_flag(int *out, int n, int m) {
    int tid = threadIdx.x;
    if (tid < n) {
        int found = -1;
        for (int i = tid; i < n && found < 0; i += blockDim.x) {
            for (int j = 0; j < m; j++) {
                if ((i * m + j) % 17 == 0) { found = i * m + j; break; }
            }
        }
        out[tid] = found;
    }
}

// ------------------------------------------------------------------
// Complex cast: float → int via PTX-level truncation.

__global__ void float_int_cast(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // (int)v → truncation toward zero; (unsigned)(int)v → reinterpret as unsigned
        int i = (int)v;
        unsigned u = (unsigned)i;
        out[tid] = (int)(u ^ (unsigned)i);
    }
}

// ------------------------------------------------------------------
// 64-bit arithmetic boundary: INT64_MIN / INT64_MAX patterns.

__global__ void int64_boundary(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        long long abs_v = v < 0 ? -v : v;
        // Saturate: INT64_MAX = 9223372036854775807
        if (abs_v > 9223372036854775807LL) abs_v = 9223372036854775807LL;
        out[tid] = abs_v;
    }
}

// ------------------------------------------------------------------
// unsigned long long arithmetic with overflow patterns.

__global__ void ull_overflow(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        // Modular unsigned arithmetic — overflows wrap
        unsigned long long a = v * v;
        unsigned long long b = a + v + 1ULL;
        unsigned long long c = b - a;  // = v + 1
        out[tid] = c;
    }
}

// ------------------------------------------------------------------
// int8 arithmetic with explicit casting (simulated satmath).

__global__ void int8_sat_arith(signed char *out, signed char *a, signed char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = (int)a[tid];
        int bv = (int)b[tid];
        // Saturating multiply: clamp to [-128, 127]
        int prod = av * bv;
        prod = prod > 127 ? 127 : (prod < -128 ? -128 : prod);
        out[tid] = (signed char)prod;
    }
}

// ------------------------------------------------------------------
// uint16 arithmetic and masking.

__global__ void uint16_arith(unsigned short *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned short v = in[tid];
        // Modular 16-bit: (v*3 + 1) & 0xFFFF
        unsigned int t = (unsigned int)v * 3u + 1u;
        out[tid] = (unsigned short)(t & 0xFFFFu);
    }
}
