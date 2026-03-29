// Probe: previously-broken arc-trig functions — atanf, asinf, acosf,
// and various combinations in real-world formulas.

// ------------------------------------------------------------------
// atanf.

__global__ void atan_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = atanf(in[tid]);
    }
}

// ------------------------------------------------------------------
// asinf.

__global__ void asin_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Clamp to [-1, 1] to avoid domain error
        v = (v > 1.0f) ? 1.0f : (v < -1.0f) ? -1.0f : v;
        out[tid] = asinf(v);
    }
}

// ------------------------------------------------------------------
// acosf.

__global__ void acos_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        v = (v > 1.0f) ? 1.0f : (v < -1.0f) ? -1.0f : v;
        out[tid] = acosf(v);
    }
}

// ------------------------------------------------------------------
// atan2f and combined trig (spherical coordinates conversion).

__global__ void spherical_angles(float *theta, float *phi,
                                   float *x, float *y, float *z, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float xv = x[tid], yv = y[tid], zv = z[tid];
        float r = hypotf(hypotf(xv, yv), zv);
        if (r > 0.0f) {
            theta[tid] = acosf(zv / r);
            phi[tid]   = atan2f(yv, xv);
        } else {
            theta[tid] = 0.0f;
            phi[tid]   = 0.0f;
        }
    }
}

// ------------------------------------------------------------------
// Law of cosines: angle from three side lengths.

__global__ void law_of_cosines(float *angle, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid], cv = c[tid];
        // angle C = acos((a^2 + b^2 - c^2) / (2*a*b))
        float cos_c = (av*av + bv*bv - cv*cv) / (2.0f * av * bv);
        cos_c = (cos_c > 1.0f) ? 1.0f : (cos_c < -1.0f) ? -1.0f : cos_c;
        angle[tid] = acosf(cos_c);
    }
}

// ------------------------------------------------------------------
// Inverse-trig used in conditional.

__global__ void safe_asin(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r;
        if (v >= -1.0f && v <= 1.0f) {
            r = asinf(v);
        } else {
            r = (v > 0.0f) ? 1.5707963f : -1.5707963f;  // pi/2 or -pi/2
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// atan used in loop for phase unwrapping (simplified).

__global__ void phase_unwrap(float *out, float *re, float *im, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float r = re[tid], i = im[tid];
        float phase = atan2f(i, r);
        // Simple unwrap: keep in [-pi, pi]
        float pi = 3.14159265f;
        while (phase > pi)  phase -= 2.0f * pi;
        while (phase < -pi) phase += 2.0f * pi;
        out[tid] = phase;
    }
}
