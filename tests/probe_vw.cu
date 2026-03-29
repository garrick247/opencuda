// Probe: device math builtins — abs/fabs, min/max, sqrt/rsqrt, exp/log/pow,
// floor/ceil/round/trunc, fmod, and sinf/cosf. Also tests mixed signed/unsigned
// comparison behavior.

#include <math.h>

// ------------------------------------------------------------------
// abs() and fabs() / fabsf().

__global__ void abs_variants(int *iout, float *fout, int *iin, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   iv = abs(iin[tid]);       // integer abs
        float fv = fabsf(fin[tid]);     // float abs
        iout[tid] = iv;
        fout[tid] = fv;
    }
}

// ------------------------------------------------------------------
// min() / max() — integer and float.

__global__ void min_max(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int lo = min(v, 100);
        int hi = max(lo, 0);
        out[tid] = hi;  // clamp(v, 0, 100)
    }
}

__global__ void fmin_fmax(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float lo = fminf(v, 1.0f);
        float hi = fmaxf(lo, 0.0f);
        out[tid] = hi;  // clamp(v, 0.0, 1.0)
    }
}

// ------------------------------------------------------------------
// sqrtf / rsqrtf.

__global__ void sqrt_rsqrt(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float s = sqrtf(v);
        float rs = rsqrtf(v);
        out[tid] = s * rs;  // sqrt(v) * (1/sqrt(v)) ≈ 1.0
    }
}

// ------------------------------------------------------------------
// expf / logf.

__global__ void exp_log(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float e = expf(v);
        float l = logf(e);   // log(exp(v)) = v (for valid inputs)
        out[tid] = l;
    }
}

// ------------------------------------------------------------------
// powf.

__global__ void pow_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = powf(v, 3.0f);   // v^3
    }
}

// ------------------------------------------------------------------
// floor / ceil / round / truncf.

__global__ void rounding(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float fl = floorf(v);
        float ce = ceilf(v);
        float ro = roundf(v);
        float tr = truncf(v);
        // Sum — each of these applied to same input
        out[tid] = fl + ce + ro + tr;
    }
}

// ------------------------------------------------------------------
// fmodf.

__global__ void fmod_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = fmodf(v, 3.14159f);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// sinf / cosf.

__global__ void sincos_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float s = sinf(v);
        float c = cosf(v);
        out[tid] = s * s + c * c;  // sin^2 + cos^2 ≈ 1.0
    }
}

// ------------------------------------------------------------------
// Mixed signed/unsigned comparison (unsigned wins by C rules).

__global__ void signed_unsigned_cmp(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int u = in[tid];
        int s = (int)u - 5;
        // Explicitly cast for comparison
        int r = ((unsigned int)s < u) ? 1 : 0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// atanf / atan2f.

__global__ void atan_kernel(float *out, float *y, float *x, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float angle = atan2f(y[tid], x[tid]);
        out[tid] = angle;
    }
}
