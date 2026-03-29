// Probe: unusual C patterns that test parser robustness — computed array
// sizes, complex struct initialization, multi-path goto, type-width
// arithmetic, and edge cases in function inlining.

// ------------------------------------------------------------------
// Computed array size via #define expression.

#define BUF_SIZE (4 * 4)
#define PAD 2
#define PADDED_SIZE (BUF_SIZE + PAD * 2)

__global__ void computed_size_array(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[PADDED_SIZE];
        for (int i = 0; i < PADDED_SIZE; i++) buf[i] = 0;
        // Write with padding
        for (int i = 0; i < BUF_SIZE; i++)
            buf[i + PAD] = in[(tid * BUF_SIZE + i) % n];
        // Sum without padding
        int s = 0;
        for (int i = PAD; i < PAD + BUF_SIZE; i++) s += buf[i];
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Complex struct initialization with all-field assignment.

struct Color {
    unsigned char r, g, b, a;
};

__device__ struct Color make_color(int v) {
    struct Color c;
    c.r = (unsigned char)(v & 0xFF);
    c.g = (unsigned char)((v >> 8) & 0xFF);
    c.b = (unsigned char)((v >> 16) & 0xFF);
    c.a = 255;
    return c;
}

__global__ void color_kernel(unsigned int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Color c = make_color(in[tid]);
        // Pack back to int
        out[tid] = (unsigned int)c.r
                 | ((unsigned int)c.g << 8)
                 | ((unsigned int)c.b << 16)
                 | ((unsigned int)c.a << 24);
    }
}

// ------------------------------------------------------------------
// Linked-list-like traversal (simulated in arrays).

__global__ void list_traverse(int *out, int *next, int *vals, int start, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int cur = start;
        int sum = 0;
        int steps = 0;
        while (cur >= 0 && cur < n && steps < n) {
            sum += vals[cur];
            cur = next[cur];
            steps++;
        }
        out[0] = sum;
        out[1] = steps;
    }
}

// ------------------------------------------------------------------
// Multiple independent data streams in same kernel.

__global__ void dual_stream(float *out_a, float *out_b,
                             float *in_a, float *in_b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Two independent reductions
        float a = in_a[tid];
        float b = in_b[tid];
        float sum_a = a + __shfl_xor_sync(0xFFFFFFFF, a, 16);
        float sum_b = b + __shfl_xor_sync(0xFFFFFFFF, b, 16);
        sum_a += __shfl_xor_sync(0xFFFFFFFF, sum_a, 8);
        sum_b += __shfl_xor_sync(0xFFFFFFFF, sum_b, 8);
        sum_a += __shfl_xor_sync(0xFFFFFFFF, sum_a, 4);
        sum_b += __shfl_xor_sync(0xFFFFFFFF, sum_b, 4);
        sum_a += __shfl_xor_sync(0xFFFFFFFF, sum_a, 2);
        sum_b += __shfl_xor_sync(0xFFFFFFFF, sum_b, 2);
        sum_a += __shfl_xor_sync(0xFFFFFFFF, sum_a, 1);
        sum_b += __shfl_xor_sync(0xFFFFFFFF, sum_b, 1);
        if ((threadIdx.x & 31) == 0) {
            out_a[tid / 32] = sum_a;
            out_b[tid / 32] = sum_b;
        }
    }
}

// ------------------------------------------------------------------
// Type-width arithmetic: mix of int, long, long long.

__global__ void type_width_arith(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int iv = in[tid];
        long ll = (long)iv * 1000L;
        long long lll = (long long)iv * 1000000LL;
        // Result: iv * 1000 + iv * 1000000
        out[tid] = (long long)ll + lll;
    }
}

// ------------------------------------------------------------------
// Reciprocal via Newton-Raphson (double precision).

__device__ double fast_rcp_dbl(double x) {
    // Initial estimate via float
    float xf = (float)x;
    float rf = 1.0f / xf;
    double r = (double)rf;
    // One NR iteration: r = r * (2 - x * r)
    r = r * (2.0 - x * r);
    // Second iteration for double precision
    r = r * (2.0 - x * r);
    return r;
}

__global__ void rcp_dbl_kernel(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fast_rcp_dbl(in[tid]);
    }
}

// ------------------------------------------------------------------
// Complex predicate expression: de Morgan's law.

__global__ void demorgan_test(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        // NOT(A AND B AND C) == NOT(A) OR NOT(B) OR NOT(C)
        int lhs = !(av && bv && cv);
        int rhs = (!av) || (!bv) || (!cv);
        out[tid] = (lhs == rhs) ? 1 : 0;  // should always be 1
    }
}

// ------------------------------------------------------------------
// Shared memory bank testing: stride-32 access pattern.

__global__ void bank_stride_32(int *out, int *in, int n) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (tid < 32 && gid < n) smem[tid] = in[gid];
    __syncthreads();

    // Read from stride-32 (same bank for all threads in half-warp — bank conflict)
    // but PTX is still valid
    if (tid < 32 && gid < n) {
        out[gid] = smem[(tid * 1) % 32];  // stride-1 (no conflict)
    }
}
