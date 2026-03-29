// Probe: math intrinsics — transcendentals, rounding, special values,
// and CUDA-specific math functions.

// ------------------------------------------------------------------
// Basic transcendentals.

__global__ void transcendentals(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = __sinf(v) + __cosf(v) + __expf(v) + __logf(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Fast math intrinsics.

__global__ void fast_math(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = __fdividef(v, v + 1.0f) + __powf(v, 2.0f) + __sqrtf(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Reciprocal sqrt and rounding.

__global__ void rounding_ops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = floorf(v) + ceilf(v) + roundf(v) + truncf(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// fmaf: fused multiply-add (single instruction).

__global__ void fma_ops(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float r = fmaf(a[tid], b[tid], c[tid]);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Double precision transcendentals.

__global__ void double_transcendentals(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double r = sin(v) + cos(v) + exp(v);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Integer math: abs, min, max.

__global__ void int_math(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = abs(v);
        int b = min(v, 100);
        int c = max(v, -100);
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Float abs, min, max.

__global__ void float_math(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = fabsf(v);
        float b = fminf(v, 100.0f);
        float c = fmaxf(v, -100.0f);
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Mixed: atan2f, hypotf, ldexpf, frexpf-style.

__global__ void mixed_math(float *out, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float xv = x[tid], yv = y[tid];
        float angle = atan2f(yv, xv);
        float hyp   = hypotf(xv, yv);
        float r = angle + hyp;
        out[tid] = r;
    }
}
