// Probe: device fn early-return paths, bit-ops on float via union,
// do-while(0) multi-statement macros, __launch_bounds__ variations,
// and complex control flow with multiple state variables.

// ------------------------------------------------------------------
// Device fn: early return on each condition (many return paths).

__device__ int multi_return_path(int v, int mode) {
    if (mode == 0) return v;
    if (mode == 1) return v * 2;
    if (mode == 2) return v * v;
    if (mode == 3) return v + 100;
    return -1;
}

__global__ void many_return_paths(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = multi_return_path(in[tid], tid % 5);
    }
}

// ------------------------------------------------------------------
// Bit operations on float via union (IEEE 754 manipulation).

union FloatBits { float f; unsigned int u; };

__global__ void float_sign_flip(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union FloatBits b;
        b.f = in[tid];
        b.u ^= 0x80000000u;   // flip sign bit
        out[tid] = b.f;       // negated float
    }
}

__global__ void float_abs_bits(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        union FloatBits b;
        b.f = in[tid];
        b.u &= 0x7FFFFFFFu;   // clear sign bit
        out[tid] = b.f;        // |float|
    }
}

// ------------------------------------------------------------------
// Multi-statement macro using do-while(0).

#define SWAP_INTS(a, b) do { int _tmp = (a); (a) = (b); (b) = _tmp; } while (0)

__global__ void swap_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int y = in[n - 1 - tid];   // mirror element
        SWAP_INTS(x, y);
        out[tid] = x + y;   // = in[n-1-tid] + in[tid] (sum unchanged)
    }
}

// ------------------------------------------------------------------
// __launch_bounds__ with min_blocks parameter.

__global__ __launch_bounds__(256, 4) void launch_bounds_2param(int *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = tid;
}

// ------------------------------------------------------------------
// Complex FSM: multiple state variables changing together.

__global__ void fsm_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int state = 0;
        int acc   = 0;
        int count = 0;
        for (int i = 0; i < 8; i++) {
            switch (state) {
                case 0:
                    acc += v;
                    if (acc > 10) state = 1;
                    break;
                case 1:
                    acc *= 2;
                    count++;
                    if (count >= 2) state = 2;
                    break;
                case 2:
                    acc -= v;
                    break;
            }
        }
        out[tid] = acc + count;
    }
}

// ------------------------------------------------------------------
// __device__ fn with pointer-to-local-struct output parameter.

struct Stats { int sum; int min_val; int max_val; };

__device__ void compute_stats(const int *arr, int len, struct Stats *s) {
    s->sum = 0;
    s->min_val = arr[0];
    s->max_val = arr[0];
    for (int i = 0; i < len; i++) {
        s->sum += arr[i];
        if (arr[i] < s->min_val) s->min_val = arr[i];
        if (arr[i] > s->max_val) s->max_val = arr[i];
    }
}

__global__ void stats_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid * 4 + 3 < n) {
        struct Stats s;
        compute_stats(in + tid * 4, 4, &s);
        out[tid] = s.sum + s.min_val + s.max_val;
    }
}

// ------------------------------------------------------------------
// Loop that modifies both index and data variable simultaneously.

__global__ void concurrent_loop_mod(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0;
        while (i < 4 && v > 0) {
            v /= 2;
            i++;
        }
        out[tid] = i;   // number of halving steps until v <= 0
    }
}

// ------------------------------------------------------------------
// Complex expression tree: deeply nested arithmetic.

__global__ void deep_expr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // (a + b) * (c - d) / (e + f) — uses v multiple times
        int a = v + 1, b = v - 1;
        int c = v * 2, d = v / 2;
        int e = v + 10, f = v - 10;
        int r = ((a + b) * (c - d)) / ((e + f > 0) ? (e + f) : 1);
        out[tid] = r;
    }
}
