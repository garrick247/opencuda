// Probe: register class edge cases — double precision in complex
// expressions, double as condition, and double/float interoperability.

// ------------------------------------------------------------------
// Double in ternary condition (setp.lt.f64).

__global__ void double_cond(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        // Ternary with double condition
        out[tid] = (v > 0.0) ? v * 2.0 : -v;
    }
}

// ------------------------------------------------------------------
// Double comparison in if chain.

__global__ void double_if(int *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        int r;
        if (v < -1.0)      r = -2;
        else if (v < 0.0)  r = -1;
        else if (v < 1.0)  r = 0;
        else if (v < 10.0) r = 1;
        else               r = 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float and double mixed in same expression (float promoted to double).

__global__ void float_double_arith(double *out, float *fa, double *da, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // fa[tid] is float, da[tid] is double
        // The mixed expression promotes float to double
        double r = fa[tid] * da[tid] + (double)fa[tid] - da[tid];
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Double accumulation in loop.

__global__ void double_accum_loop(double *out, float *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        double acc = 0.0;
        for (int i = 0; i < k; i++) {
            acc += (double)in[tid * k + i];
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Double math functions.

__global__ void double_math(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        double r = 0.0;
        if (v > 0.0) {
            r = sqrt(v);  // sqrt of double
        }
        r += fabs(v);     // fabs of double
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Double to long long and back.

__global__ void double_ll_roundtrip(long long *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        // Clamp to ll range
        if (v > 1e18) v = 1e18;
        if (v < -1e18) v = -1e18;
        long long ll = (long long)v;
        out[tid] = ll;
    }
}

// ------------------------------------------------------------------
// Complex double computation: Mandelbrot-like iteration.

__global__ void mandelbrot_iter(int *out, double *re, double *im, int n, int max_iter) {
    int tid = threadIdx.x;
    if (tid < n) {
        double cr = re[tid], ci = im[tid];
        double zr = 0.0, zi = 0.0;
        int iter = 0;
        while (iter < max_iter && zr*zr + zi*zi < 4.0) {
            double new_zr = zr*zr - zi*zi + cr;
            double new_zi = 2.0*zr*zi + ci;
            zr = new_zr;
            zi = new_zi;
            iter++;
        }
        out[tid] = iter;
    }
}
