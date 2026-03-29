// Probe: half-precision (__half) arithmetic, mixed half/float kernels.
// Uses __half* arrays directly (b16 loads/stores) to avoid u16↔f16 reinterpret.

#include <cuda_fp16.h>

// ------------------------------------------------------------------
// Half precision: add two __half arrays.

__global__ void half_add(__half *out, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = a[tid];
        __half hb = b[tid];
        out[tid] = __hadd(ha, hb);
    }
}

// ------------------------------------------------------------------
// Half multiply.

__global__ void half_mul(__half *out, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __hmul(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Half fused multiply-add.

__global__ void half_fma(__half *out, __half *a, __half *b, __half *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __hfma(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// Half/float conversion: float → half → float round-trip.

__global__ void half_cvt_roundtrip(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __float2half(in[tid]);
        out[tid] = __half2float(h);
    }
}

// ------------------------------------------------------------------
// Half comparison: return bitmask.

__global__ void half_cmp(int *out, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = 0;
        if (__hgt(a[tid], b[tid])) r |= 1;
        if (__hlt(a[tid], b[tid])) r |= 2;
        if (__heq(a[tid], b[tid])) r |= 4;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Half min/max.

__global__ void half_minmax(__half *omin, __half *omax, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        omin[tid] = __hfmin(a[tid], b[tid]);
        omax[tid] = __hfmax(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Half subtract.

__global__ void half_sub(__half *out, __half *a, __half *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __hsub(a[tid], b[tid]);
    }
}
