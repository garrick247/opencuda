// Probe: complex loop unrolling interactions —
// loop with accumulator AND index-dependent computation (can't fully fold),
// two separate unrolled loops in same kernel,
// loop with stride != 1,
// loop where body is a single store (no accumulation)

// Trip-count-5 loop: sum of i^2 for i=0..4 = 0+1+4+9+16 = 30
__global__ void sum_squares_5(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 5; i++) {
            sum += i * i;
        }
        out[0] = sum;   // should be 30 (or close, may not fully fold across blocks)
    }
}

// Two unrolled loops in sequence — second sees updated state from first
__global__ void two_loops(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < 4; i++) {   // sum 0+1+2+3 = 6
            acc += i;
        }
        for (int j = 0; j < 4; j++) {   // acc += j: 6+0+1+2+3 = 12
            acc += j;
        }
        out[0] = acc;   // should be 12
    }
}

// Loop with stride 2: for(i=0; i<10; i+=2) — trip count 5, unrollable
__global__ void stride_loop(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 10; i += 2) {
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop from N downward: for(i=7; i>=0; i--) — trip count 8
__global__ void count_down(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int product = 1;
        for (int i = 7; i >= 1; i--) {   // 7! = 5040
            product *= i;
        }
        out[0] = product;
    }
}

// Loop body with conditional — not unrollable due to complex body?
// Actually: trip count 4, no break/continue → should unroll
__global__ void loop_with_cond(int *out, int threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        for (int i = 0; i < 4; i++) {
            if (i > threshold) count++;
        }
        out[0] = count;
    }
}
