// Probe: __half arithmetic in control flow, half comparisons used
// as branch predicates, mixed half/float computations.

#include <cuda_fp16.h>

// ------------------------------------------------------------------
// Half conditional: use half comparison as branch condition.

__global__ void half_cond(float *out, __half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = in[tid];
        __half zero = __float2half(0.0f);
        float r;
        // __hgt returns 1 if v > 0, used as condition
        if (__hgt(v, zero)) {
            r = __half2float(v) * 2.0f;
        } else {
            r = __half2float(v) * (-1.0f);
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Half arithmetic in loop.

__global__ void half_loop(float *out, __half *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = in[tid];
        __half acc = __float2half(0.0f);
        __half two = __float2half(2.0f);
        for (int i = 0; i < 4; i++) {
            acc = __hadd(acc, v);
            v = __hmul(v, two);  // doubles each iteration
        }
        out[tid] = __half2float(acc);
    }
}

// ------------------------------------------------------------------
// Half load/store with index computation.

__global__ void half_indexed(float *out, __half *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        if (i >= 0 && i < n) {
            float v = __half2float(in[i]);
            out[tid] = v * v;
        } else {
            out[tid] = 0.0f;
        }
    }
}

// ------------------------------------------------------------------
// Mixed half/float: half input, float computation, float output.

__global__ void half_to_float_compute(float *out, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = __half2float(a[tid]);
        float bv = __half2float(b[tid]);
        // Compute in float precision
        float r = av * bv + av + bv;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float to half conversion with rounding stress.

__global__ void float_to_half_stress(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Round-trip: float → half → float
        __half h = __float2half(v);
        float back = __half2float(h);
        // Round-trip twice
        __half h2 = __float2half(back * 1.5f);
        float back2 = __half2float(h2);
        out[tid] = back + back2;
    }
}

// ------------------------------------------------------------------
// Half FMA.

__global__ void half_fma_kernel(float *out, __half *a, __half *b, __half *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half r = __hfma(a[tid], b[tid], c[tid]);
        out[tid] = __half2float(r);
    }
}
