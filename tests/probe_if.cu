// Probe: Correctness-sensitive patterns —
// loop-carried dependencies that could be broken by CSE/LICM,
// phi-node updates across loop iterations,
// conditional updates with multiple possible values

// Running maximum — loop carries max across iterations
__global__ void running_max(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mx = in[0];
        for (int i = 1; i < n; i++) {
            if (in[i] > mx) mx = in[i];
        }
        *out = mx;
    }
}

// Fibonacci-like: two loop-carried vars
__global__ void fib_like(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 1;
        for (int i = 0; i < n; i++) {
            int c = a + b;
            a = b;
            b = c;
        }
        *out = a;
    }
}

// Loop where index is used BOTH as loop var and as array index
// (CSE must not merge the loop-variable register with the array offset)
__global__ void index_dual_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i] + i;  // i used as value AND as index
        }
        *out = sum;
    }
}

// Conditional accumulate: the value conditionally added varies each iteration
__global__ void cond_accum(float *out, float *in, float threshold, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float pos_sum = 0.0f;
        float neg_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = in[i];
            if (v >= threshold) {
                pos_sum += v;
            } else {
                neg_sum += v;
            }
        }
        out[0] = pos_sum;
        out[1] = neg_sum;
    }
}
