// Probe: loop unrolling interaction with conditionals inside loop body
// Also: loop with non-trivial condition (not just i < N)

__global__ void unroll_with_branch(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            float v = in[(tid + i) % n];
            if (v > 0.0f) {
                sum += v;
            } else {
                sum -= v;
            }
        }
        out[tid] = sum;
    }
}

// Loop with early exit (non-trivial condition)
__global__ void loop_early_exit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int count = 0;
        for (int i = tid; i < n && count < 8; i += 4) {
            if (in[i] > 0) count++;
        }
        out[tid] = count;
    }
}

// Loop unrolling: exactly 8 iterations (within unroll limit)
__global__ void unroll_8(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float prod = 1.0f;
        for (int i = 0; i < 8; i++) {
            prod *= in[(tid + i) % n];
        }
        out[tid] = prod;
    }
}

// Loop NOT unrolled: 17 iterations (above limit of 16)
__global__ void no_unroll_17(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 17; i++) {
            sum += in[(tid + i) % n];
        }
        out[tid] = sum;
    }
}
