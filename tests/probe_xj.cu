// Probe: complex macro patterns with function-like expansion, loop edge
// cases (unroll count exactly at trip limit), const global arrays,
// and multiple device functions calling each other (call chain depth).

// ------------------------------------------------------------------
// Macro generating code via function-like expansion.

#define SWAP(T, a, b) do { T _tmp = (a); (a) = (b); (b) = _tmp; } while(0)
#define CLAMP_TO(val, lo, hi) ((val) < (lo) ? (lo) : (val) > (hi) ? (hi) : (val))
#define ABS_DIFF(a, b) ((a) > (b) ? (a) - (b) : (b) - (a))

__global__ void macro_suite(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = in[(tid + 1) % n];
        SWAP(int, a, b);
        int c = CLAMP_TO(a, -10, 10);
        int d = ABS_DIFF(a, b);
        out[tid] = c + d;
    }
}

// ------------------------------------------------------------------
// __constant__ array: LUT lookup.

__constant__ float c_sin_table[8];  // sin(0), sin(pi/4), ..., sin(7pi/4)

__global__ void const_lut_lookup(float *out, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid] & 7;
        out[tid] = c_sin_table[idx];
    }
}

// ------------------------------------------------------------------
// __constant__ struct.

struct Constants {
    float alpha;
    float beta;
    int   max_iter;
};

__constant__ struct Constants c_params;

__global__ void use_const_struct(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int iter = 0;
        while (fabsf(v) > c_params.alpha && iter < c_params.max_iter) {
            v = v * c_params.beta;
            iter++;
        }
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Deep call chain: d4 calls d3 calls d2 calls d1.

__device__ int d1(int x) { return x ^ (x >> 1); }
__device__ int d2(int x) { return d1(x) + d1(x >> 1); }
__device__ int d3(int x) { return d2(x) * d2(x - 1); }
__device__ int d4(int x) { return d3(x) ^ d3(x + 1); }

__global__ void deep_call_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = d4(in[tid]);
    }
}

// ------------------------------------------------------------------
// Loop unroll at exactly trip limit (16): should be unrolled.

__global__ void unroll_at_limit(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += in[(tid + i) % n];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Loop unroll just over limit (17): should stay as loop.

__global__ void no_unroll_over_limit(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 17; i++) {
            sum += in[(tid + i) % n];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Conditional compile-time constant: #if chain.

#define MODE 2

#if MODE == 1
#define PROCESS(x) ((x) * 2)
#elif MODE == 2
#define PROCESS(x) ((x) * (x))
#else
#define PROCESS(x) ((x) + 1)
#endif

__global__ void conditional_define(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = PROCESS(in[tid]);  // MODE=2: x*x
    }
}

// ------------------------------------------------------------------
// Mixed global and shared memory in same access pattern.

__device__ int g_multiplier[4] = {1, 2, 4, 8};  // not yet initialized by GPU

__global__ void mixed_global_shared(int *out, int *in, int n) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (tid < 32 && gid < n) {
        smem[tid] = in[gid];
    }
    __syncthreads();

    if (gid < n) {
        // Use smem value * global multiplier based on lane
        int lane = tid & 3;
        int mult = g_multiplier[lane];
        out[gid] = smem[tid & 31] * mult;
    }
}

// ------------------------------------------------------------------
// Function with many local variables (test register pressure).

__device__ float poly10(float x) {
    float x2  = x * x;
    float x3  = x2 * x;
    float x4  = x3 * x;
    float x5  = x4 * x;
    float x6  = x5 * x;
    float x7  = x6 * x;
    float x8  = x7 * x;
    float x9  = x8 * x;
    float x10 = x9 * x;
    // Horner: 1 + x + x^2/2! + ... (first 10 terms of e^x Taylor series)
    return 1.0f + x + x2/2.0f + x3/6.0f + x4/24.0f + x5/120.0f
                + x6/720.0f + x7/5040.0f + x8/40320.0f + x9/362880.0f
                + x10/3628800.0f;
}

__global__ void poly_approx(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = poly10(in[tid]);
    }
}
