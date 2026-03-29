// Probe: pointer arithmetic stride correctness for float/double arrays,
// float-double implicit promotion, double arithmetic output,
// pointer difference, const array access pattern

// Float array: stride should be 4 bytes per element
__global__ void float_ptr_arith(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Access in[tid], in[tid+1], in[tid+2] via pointer arithmetic
        float *p = in + tid;
        float a = p[0];
        float b = p[1];
        float c = p[2];
        out[tid] = a + b + c;
    }
}

// Double array: stride should be 8 bytes per element
__global__ void double_ptr_arith(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double *p = in + tid;
        double a = p[0];
        double b = p[1];
        out[tid] = a * b;
    }
}

// Float-to-double implicit promotion in mixed arithmetic
__global__ void float_double_promo(double *out, float *fin, double *din, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = fin[tid];
        double d = din[tid];
        // float + double should promote float to double
        double result = f + d;
        out[tid] = result;
    }
}

// int + float → float promotion
__global__ void int_float_promo(float *out, int *iin, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = iin[tid];
        float f = fin[tid];
        float result = i + f;   // int promoted to float
        out[tid] = result;
    }
}

// Pointer to mid-array: base + constant offset
__global__ void mid_array_ptr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int *mid = in + n / 2;   // points to middle of array
        int sum = 0;
        for (int i = 0; i < n / 2; i++) {
            sum += mid[i];
        }
        out[0] = sum;
    }
}
