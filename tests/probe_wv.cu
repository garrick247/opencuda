// Probe: printf format strings with more types, nested device fn calls,
// multi-level recursion (iterative), complex predicate expressions,
// and array-of-structs vs struct-of-arrays patterns.

// ------------------------------------------------------------------
// printf with multiple format types.

__global__ void printf_formats(int *in, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && tid < n) {
        int   iv = in[0];
        float fv = fin[0];
        // Test various format specifiers
        printf("int=%d float=%.2f uint=%u hex=0x%x\n",
               iv, fv, (unsigned int)iv, iv);
    }
}

// ------------------------------------------------------------------
// Nested device fn calls: f3(f2(f1(x))).

__device__ int f1(int x) { return x * 2 + 1; }
__device__ int f2(int x) { return x * x - x; }
__device__ int f3(int x) { return (x >> 1) ^ x; }

__global__ void nested_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = f3(f2(f1(in[tid])));
    }
}

// ------------------------------------------------------------------
// Complex boolean predicate: (a > 0) && (b < 10) || (c == 5).

__global__ void complex_pred(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        int cond = (av > 0 && bv < 10) || (cv == 5);
        out[tid] = cond ? av + bv : cv;
    }
}

// ------------------------------------------------------------------
// Struct-of-arrays layout (SoA).

__global__ void soa_transform(float *xs, float *ys, float *zs,
                               float *out_x, float *out_y, float *out_z,
                               float scale, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out_x[tid] = xs[tid] * scale;
        out_y[tid] = ys[tid] * scale;
        out_z[tid] = zs[tid] * scale;
    }
}

// ------------------------------------------------------------------
// Array-of-structs layout (AoS).

struct Vec3 { float x, y, z; };

__global__ void aos_transform(struct Vec3 *pts, float scale, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        pts[tid].x *= scale;
        pts[tid].y *= scale;
        pts[tid].z *= scale;
    }
}

// ------------------------------------------------------------------
// String in printf (no format args).

__global__ void printf_string(int *flag, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && tid < n && flag[0]) {
        printf("hello from GPU\n");
    }
}

// ------------------------------------------------------------------
// Device function returning unsigned.

__device__ unsigned int pack_byte(unsigned char hi, unsigned char lo) {
    return ((unsigned int)hi << 8) | (unsigned int)lo;
}

__global__ void pack_bytes(unsigned int *out, unsigned char *hi, unsigned char *lo, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = pack_byte(hi[tid], lo[tid]);
    }
}

// ------------------------------------------------------------------
// Signed saturation clamp without branch.

__device__ int sat_clamp(int v, int lo, int hi) {
    int r = v;
    if (r < lo) r = lo;
    if (r > hi) r = hi;
    return r;
}

__global__ void clamp_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sat_clamp(in[tid], -100, 100);
    }
}

// ------------------------------------------------------------------
// Multiple output parameters via pointers.

__device__ void divmod(int a, int b, int *quot, int *rem) {
    *quot = a / b;
    *rem  = a % b;
}

__global__ void divmod_kernel(int *q_out, int *r_out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int q, r;
        divmod(a[tid], b[tid], &q, &r);
        q_out[tid] = q;
        r_out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Mixed predicate and arithmetic: compute max(a, b) without branch.

__global__ void branchless_max(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        // max without ternary: a + ((b - a) & ((b - a) >> 31))
        // Actually just use the ternary — test that it generates predicated selp
        out[tid] = (av > bv) ? av : bv;
    }
}

// ------------------------------------------------------------------
// Iterative Fibonacci (no recursion, loop-based).

__global__ void fib_kernel(int *out, int *n_arr, int count) {
    int tid = threadIdx.x;
    if (tid < count) {
        int n = n_arr[tid];
        if (n <= 0) { out[tid] = 0; return; }
        if (n == 1) { out[tid] = 1; return; }
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            int c = a + b;
            a = b;
            b = c;
        }
        out[tid] = b;
    }
}
