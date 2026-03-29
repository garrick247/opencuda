// Probe: loop unroller edge cases — zero-trip loops, single-trip,
// trip count determined by #define, nested unrollable loops.

#define ITERS 8
#define HALF  4

// ------------------------------------------------------------------
// Zero-trip loop: loop body never executes (bound = 0).

__global__ void zero_trip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = in[tid];
        for (int i = 0; i < 0; i++) {
            acc += i;   // never executes
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Single-trip loop: exactly one iteration.

__global__ void single_trip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 1; i++) {
            acc = in[tid] + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Unroll trip count from #define.

__global__ void define_trip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < ITERS; i++) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Two separate unrollable loops in sequence, with same bound.

__global__ void two_unroll_loops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum1 = 0, sum2 = 0;
        for (int i = 0; i < HALF; i++) {
            sum1 += v + i;
        }
        for (int i = 0; i < HALF; i++) {
            sum2 += v - i;
        }
        out[tid] = sum1 + sum2;
    }
}

// ------------------------------------------------------------------
// Nested unrollable loops: inner and outer both unrollable.

__global__ void nested_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                acc += v * i + j;
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop with stride > 1: for(i=0; i<8; i+=2).

__global__ void stride_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i += 2) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop-carried float accumulator — must fold correctly after unrolling.

__global__ void float_unroll(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float acc = 0.0f;
        for (int i = 0; i < 4; i++) {
            acc = acc + v * (float)i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Unrollable loop updating two arrays (forces correct remapping).

__global__ void dual_array_unroll(float *out1, float *out2, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = 0.0f, b = 1.0f;
        for (int i = 0; i < 4; i++) {
            a = a + v;
            b = b * v;
        }
        out1[tid] = a;
        out2[tid] = b;
    }
}
