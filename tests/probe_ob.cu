// Probe: volatile memory, __ldg, double-precision math, fma
// Tests areas not yet exercised by existing probes.

// ------------------------------------------------------------------
// volatile load/store: must emit ld.volatile.global / st.volatile.global
// (not ld.global or st.global which may be reordered by hardware).

__global__ void volatile_rw(int *out, volatile int *flag, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = flag[tid];    // volatile load
        flag[tid] = v + 1;    // volatile store
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// __ldg: read-only texture cache load.
// Must emit ld.global.nc (same as const __restrict__, but via __ldg intrinsic).

__global__ void ldg_load(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = __ldg(&data[tid]);
        out[tid] = v * 3.0f;
    }
}

// ------------------------------------------------------------------
// Double precision arithmetic: mul, add with double literal, negation.
// Must emit mul.f64, add.f64, neg.f64 — not f32 variants.

__global__ void dp_math(double *out, double *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = data[tid];
        out[tid*4+0] = v * v;          // mul.f64
        out[tid*4+1] = v + 1.5;        // add.f64 with double literal
        out[tid*4+2] = -v;             // neg.f64
        out[tid*4+3] = v - 0.5;        // sub.f64
    }
}

// ------------------------------------------------------------------
// Accumulate with multiply-then-add (float).
// Tests that mul.f32 + add.f32 are emitted correctly.

__global__ void fma_accum(float *out, float *a, float *b, float c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float result = a[tid] * b[tid] + c;
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Integer min/max via ternary (no intrinsic).
// Tests that conditional + phi-merge correctly handles signed comparisons.

__global__ void int_minmax(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int mn = (v < -100) ? -100 : v;
        int mx = (mn > 100) ? 100 : mn;
        out[tid] = mx;
    }
}
