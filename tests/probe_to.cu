// Probe: previously-missing math functions — powf, hypotf, atan2f, cbrtf.

// ------------------------------------------------------------------
// powf(x, y).

__global__ void pow_kernel(float *out, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = powf(x[tid], y[tid]);
    }
}

// ------------------------------------------------------------------
// hypotf(a, b).

__global__ void hypot_kernel(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = hypotf(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// atan2f(y, x).

__global__ void atan2_kernel(float *out, float *y, float *x, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = atan2f(y[tid], x[tid]);
    }
}

// ------------------------------------------------------------------
// cbrtf(x).

__global__ void cbrt_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = cbrtf(in[tid]);
    }
}

// ------------------------------------------------------------------
// Combined: polar to cartesian using atan2f, hypotf.

__global__ void polar_ops(float *rout, float *aout, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        rout[tid] = hypotf(x[tid], y[tid]);
        aout[tid] = atan2f(y[tid], x[tid]);
    }
}

// ------------------------------------------------------------------
// Power law: pow used in expression.

__global__ void power_law(float *out, float *in, float exponent, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        if (v > 0.0f) {
            out[tid] = powf(v, exponent);
        } else {
            out[tid] = 0.0f;
        }
    }
}

// ------------------------------------------------------------------
// Geometric mean using cbrt and pow.

__global__ void geomean3(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float product = a[tid] * b[tid] * c[tid];
        if (product > 0.0f) {
            out[tid] = cbrtf(product);
        } else {
            out[tid] = 0.0f;
        }
    }
}
