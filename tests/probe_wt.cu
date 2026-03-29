// Probe: complex arithmetic expressions, multi-dimensional shared memory,
// register reuse patterns, type-punning via union, and edge cases in
// integer/float promotion.

// ------------------------------------------------------------------
// Horner's method polynomial evaluation.

__device__ float horner5(float x, float a0, float a1, float a2, float a3, float a4) {
    // ((((a4*x + a3)*x + a2)*x + a1)*x + a0)
    float r = a4;
    r = r * x + a3;
    r = r * x + a2;
    r = r * x + a1;
    r = r * x + a0;
    return r;
}

__global__ void poly_eval(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = horner5(x, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
    }
}

// ------------------------------------------------------------------
// 2D shared memory tile: each element is row*COLS + col.

#define TILE 8
__global__ void shared_2d_tile(int *out, int n) {
    __shared__ int tile[TILE][TILE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gid = blockIdx.x * (TILE * TILE) + ty * TILE + tx;

    tile[ty][tx] = ty * TILE + tx;
    __syncthreads();

    // Read transpose
    if (gid < n) {
        out[gid] = tile[tx][ty];
    }
}

// ------------------------------------------------------------------
// Union type pun: reinterpret int bits as float without cast intrinsic.

union IntFloat {
    int   i;
    float f;
};

__global__ void union_pun(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union IntFloat u;
        u.i = in[tid];
        out[tid] = u.f;
    }
}

// ------------------------------------------------------------------
// Union with array member.

union ByteWord {
    unsigned int  word;
    unsigned char bytes[4];
};

__global__ void union_bytes(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union ByteWord bw;
        bw.word = in[tid];
        // Byte swap
        unsigned char b0 = bw.bytes[0];
        unsigned char b1 = bw.bytes[1];
        unsigned char b2 = bw.bytes[2];
        unsigned char b3 = bw.bytes[3];
        bw.bytes[0] = b3;
        bw.bytes[1] = b2;
        bw.bytes[2] = b1;
        bw.bytes[3] = b0;
        out[tid] = bw.word;
    }
}

// ------------------------------------------------------------------
// Double-precision Kahan summation.

__global__ void kahan_sum(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double sum = 0.0;
        double comp = 0.0;
        // Each thread processes a strided segment
        for (int i = tid; i < n; i += blockDim.x) {
            double y = in[i] - comp;
            double t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Integer GCD via Euclid's algorithm.

__device__ int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void gcd_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = gcd(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Bit interleaving (Morton code for 2D): interleave bits of x and y.

__device__ unsigned int morton_2d(unsigned int x, unsigned int y) {
    // Spread bits of x into even positions
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    // Spread bits of y into odd positions
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
    return x | (y << 1);
}

__global__ void morton_kernel(unsigned int *out, unsigned int *xs, unsigned int *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = morton_2d(xs[tid], ys[tid]);
    }
}

// ------------------------------------------------------------------
// Parallel merge step (odd-even merge network for 8 elements).

__global__ void compare_swap(int *data, int i, int j, int n) {
    int tid = threadIdx.x;
    if (tid < n && i < n && j < n) {
        if (tid == 0) {
            if (data[i] > data[j]) {
                int tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
    }
}

// ------------------------------------------------------------------
// Float reciprocal with Newton-Raphson refinement.

__device__ float fast_rcp(float x) {
    float r = __frcp_rn(x);        // hardware reciprocal estimate
    // One NR step: r = r * (2 - x*r)
    r = r * (2.0f - x * r);
    return r;
}

__global__ void rcp_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fast_rcp(in[tid]);
    }
}

// ------------------------------------------------------------------
// __frcp_rd, __fsqrt_rn variants.

__global__ void rounding_variants(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = __fsqrt_rn(v);
        float b = __frcp_rz(v);
        out[tid] = a + b;
    }
}
