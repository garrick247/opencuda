// Probe: float register liveness edge cases.
// Float loop-carried variables, float/int register interleaving,
// float conditionals with multiple live vars, double precision liveness.

// ------------------------------------------------------------------
// 4 float loop-carried variables: all must survive each iteration.

__global__ void float4_carried(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            s0 += v;
            s1 += v * v;
            s2 += v * v * v;
            s3 += v / (float)(i + 1);
        }
        out[0] = s0 + s1 + s2 + s3;
    }
}

// ------------------------------------------------------------------
// Interleaved int and float loop-carried vars.
// Tests that int/float register classes don't alias.

__global__ void int_float_interleaved(float *out, int *idata, float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int   isum = 0;
        float fsum = 0.0f;
        int   icount = 0;
        float fmax   = -1e30f;
        for (int i = 0; i < n; i++) {
            isum   += idata[i];
            fsum   += fdata[i];
            icount += 1;
            if (fdata[i] > fmax) fmax = fdata[i];
        }
        out[0] = fsum + (float)isum + fmax + (float)icount;
    }
}

// ------------------------------------------------------------------
// Float conditional writeback: min/max/sum all in same loop.

__global__ void float_minmaxsum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && n > 0) {
        float mn = data[0], mx = data[0], sum = data[0];
        for (int i = 1; i < n; i++) {
            float v = data[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        out[0] = mn;
        out[1] = mx;
        out[2] = sum;
    }
}

// ------------------------------------------------------------------
// Double precision loop-carried variables.
// 2 double accumulators must survive each iteration.

__global__ void double_dual_accum(double *out, double *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < n; i++) {
            double v = data[i];
            sum    += v;
            sum_sq += v * v;
        }
        out[0] = sum;
        out[1] = sum_sq;
    }
}

// ------------------------------------------------------------------
// Float + double mixed loop: both must be live simultaneously.

__global__ void float_double_loop(float *fout, double *dout,
                                   float *fdata, double *ddata, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float  fsum = 0.0f;
        double dsum = 0.0;
        for (int i = 0; i < n; i++) {
            fsum += fdata[i];
            dsum += ddata[i];
        }
        fout[0] = fsum;
        dout[0] = dsum;
    }
}

// ------------------------------------------------------------------
// Float loop with early exit: `if (v > threshold) break`.
// Both `sum` and loop index must survive to exit point.

__global__ void float_break_loop(float *out, float *data, int n, float threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            if (v > threshold) break;
            sum += v;
            count++;
        }
        out[0] = sum;
        out[1] = (float)count;
    }
}

// ------------------------------------------------------------------
// Float value defined on both branches of conditional, used after.
// Tests that float phi-like merge is handled correctly.

__global__ void float_branch_merge(float *out, float *data, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        float r;
        if (flags[tid]) {
            r = v * 2.0f;
        } else {
            r = v * 0.5f;
        }
        out[tid] = r + 1.0f;
    }
}

// ------------------------------------------------------------------
// Float reduction then cast to int.
// Tests float accumulator live range followed by CVT.

__global__ void float_reduce_cast(int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        out[0] = (int)sum;  // CVT after loop: sum still live
    }
}
