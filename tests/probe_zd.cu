// Probe: remaining gaps — double comparison (all 6), double ternary,
// double-precision loop with float conversion, __double2ll_rn/__ll2double_rn,
// mixed double/int arithmetic, double fabs/fmin/fmax, atomicAdd(double*) in
// reduction pattern, double-precision dot product, and double sqrt/rsqrt.

// ------------------------------------------------------------------
// Double comparisons: all 6 ops.

__global__ void double_cmp(int *out, double *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double x = a[tid], y = b[tid];
        int bits = 0;
        if (x <  y) bits |= 1;
        if (x >  y) bits |= 2;
        if (x <= y) bits |= 4;
        if (x >= y) bits |= 8;
        if (x == y) bits |= 16;
        if (x != y) bits |= 32;
        out[tid] = bits;
    }
}

// ------------------------------------------------------------------
// Double ternary select.

__global__ void double_ternary(double *out, double *a, double *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (sel[tid] > 0) ? a[tid] : b[tid];
    }
}

// ------------------------------------------------------------------
// Double accumulator with float input conversion.

__global__ void double_from_float(double *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double s = 0.0;
        for (int i = 0; i < 32; i++) {
            s += (double)in[(tid * 32 + i) % n];
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// __double2ll_rn / __ll2double_rn.

__global__ void d2ll_ll2d(long long *out_ll, double *out_d,
                            double *in_d, long long *in_ll, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_ll[tid] = __double2ll_rn(in_d[tid]);
        out_d[tid]  = __ll2double_rn(in_ll[tid]);
    }
}

// ------------------------------------------------------------------
// Mixed double/int arithmetic.

__global__ void double_int_mix(double *out, int *in_i, double *in_d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int    iv = in_i[tid];
        double dv = in_d[tid];
        // int + double → double
        double r = (double)iv + dv;
        // int * double → double
        r += (double)iv * dv;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Double fabs / fmin / fmax.

__global__ void double_minmaxabs(double *out_abs, double *out_min, double *out_max,
                                    double *a, double *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_abs[tid] = fabs(a[tid]);
        out_min[tid] = fmin(a[tid], b[tid]);
        out_max[tid] = fmax(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// atomicAdd(double*) in reduction.

__global__ void double_reduce(double *out, double *in, int n) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// ------------------------------------------------------------------
// Double-precision dot product.

__global__ void double_dot(double *out, double *a, double *b, int n) {
    __shared__ double smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? a[gid] * b[gid] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// ------------------------------------------------------------------
// Double sqrt / rsqrt (approximation).

__global__ void double_sqrt_rsqrt(double *out_sqrt, double *out_rsqrt,
                                     double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_sqrt[tid]  = sqrt(in[tid]);
        out_rsqrt[tid] = 1.0 / sqrt(in[tid]);
    }
}

// ------------------------------------------------------------------
// Double-precision Kahan summation.

__global__ void kahan_sum(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double sum = 0.0;
        double c = 0.0;
        for (int i = 0; i < n; i++) {
            double y = in[i] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        out[tid] = sum;
    }
}
