// Probe: device function returning float, called multiple times with same name,
// double-precision arithmetic and type mixing,
// pointer-typed return from device function,
// loop with device function call that has side effects via out-pointer

// Device function returning float with computation
__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Multiple calls to same device function — each must produce distinct results
__global__ void multi_lerp(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid];
        float r0 = lerp(av, bv, 0.0f);
        float r1 = lerp(av, bv, 0.5f);
        float r2 = lerp(av, bv, 1.0f);
        out[tid * 3]     = r0;
        out[tid * 3 + 1] = r1;
        out[tid * 3 + 2] = r2;
    }
}

// Double-precision arithmetic
__global__ void double_accumulate(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
        }
        *out = sum;
    }
}

// Mixed float/double: computation upgrades to double
__global__ void mixed_precision(double *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += (double)in[i];  // widen to double then accumulate
        }
        *out = acc;
    }
}

// Device function called in loop, result accumulated
__device__ int square(int x) {
    return x * x;
}

__global__ void sum_squares(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            total += square(in[i]);
        }
        *out = total;
    }
}
