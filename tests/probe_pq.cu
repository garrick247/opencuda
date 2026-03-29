// Probe: float/double arithmetic edge cases.
// fma, mixed float/double expressions, float comparison chains,
// float ternary, float loop accumulation, double-to-float cast.

// ------------------------------------------------------------------
// Fused multiply-add: a*b + c in a loop.
// Tests that the compiler doesn't break FMA by inserting extra CVTs.

__global__ void fma_loop(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float r = fmaf(a[tid], b[tid], c[tid]);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float accumulator in a loop.
// Tests that loop-carried float variable survives writeback.

__global__ void float_accum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Mixed float and double in same kernel.
// Tests that f32/f64 registers don't alias.

__global__ void float_double_mix(double *out, float *fa, double *db, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float  fv = fa[tid];
        double dv = db[tid];
        // Widen float to double, do double arithmetic
        double r = (double)fv * dv + 1.0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float comparison chain: min of three values.
// Tests predicate generation for float comparisons.

__global__ void float_min3(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float va = a[tid], vb = b[tid], vc = c[tid];
        float m = va;
        if (vb < m) m = vb;
        if (vc < m) m = vc;
        out[tid] = m;
    }
}

// ------------------------------------------------------------------
// Float ternary: abs value via ternary.
// Tests f32 ternary with predicate select.

__global__ void float_ternary_abs(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        out[tid] = (v >= 0.0f) ? v : -v;
    }
}

// ------------------------------------------------------------------
// Double accumulator loop.
// 64-bit float register must survive loop writeback.

__global__ void double_accum(double *out, double *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Float division and reciprocal.
// Tests that float div doesn't emit integer div.

__global__ void float_div(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float denom = b[tid];
        float r = (denom != 0.0f) ? a[tid] / denom : 0.0f;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Mixed int/float: int index scaled by float factor.
// Tests that int-to-float conversion is emitted correctly.

__global__ void int_to_float_scale(float *out, int *data, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = (float)data[tid] * scale;
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Float comparison with loop-carried min/max.
// Both min_val and max_val must be live across each iteration.

__global__ void float_minmax(float *out_min, float *out_max,
                              float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float min_val = data[0];
        float max_val = data[0];
        for (int i = 1; i < n; i++) {
            float v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        out_min[0] = min_val;
        out_max[0] = max_val;
    }
}
