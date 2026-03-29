// Probe: multiple independently unrollable loops in same kernel,
// loop followed by conditional, conditional followed by loop,
// register reuse across sequential unrolled loops,
// loop where accumulator type differs from element type

// Three consecutive unrollable loops in same kernel
__global__ void three_loops(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0;
        for (int i = 0; i < 4; i++) a += i;   // 0+1+2+3=6

        int b = 1;
        for (int j = 0; j < 4; j++) b *= (j + 1);  // 1*2*3*4=24

        int c = 0;
        for (int k = 0; k < 4; k++) c += k * k;    // 0+1+4+9=14

        out[0] = a;
        out[1] = b;
        out[2] = c;
    }
}

// Loop then conditional
__global__ void loop_then_cond(int *out, int *in, int n, int threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
        }
        // Conditional after loop
        if (sum > threshold) {
            out[0] = sum;
            out[1] = 1;
        } else {
            out[0] = 0;
            out[1] = 0;
        }
    }
}

// Conditional then loop
__global__ void cond_then_loop(int *out, int *in, int n, int flag) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int base = (flag > 0) ? 10 : -10;
        int sum = base;
        for (int i = 0; i < n; i++) {
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop with float accumulator but int array input
__global__ void float_acc_of_ints(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += (float)in[i];   // int-to-float cast in body
        }
        out[0] = sum;
    }
}

// Loop accumulating into double
__global__ void double_acc_of_floats(double *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (double)in[i];
        }
        out[0] = sum;
    }
}
