// Probe: loop that starts at non-zero init — should NOT be unrolled
// (induction variable wouldn't map to iteration index),
// loop with LE condition,
// loop over runtime-initialized accumulator

// Forward loop starting at non-zero: for(i=2; i<7; i++) — trip 5
// Unroller assumes i starts at 0, so this should NOT unroll (or correctly handle)
__global__ void nonzero_start(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 2; i < 7; i++) {   // sum in[2]+in[3]+...+in[6]
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop with i <= N condition: for(i=0; i<=4; i++) — trip 5
__global__ void loop_le(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i <= 4; i++) {   // sum in[0]..in[4]
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop where accumulator is pre-initialized from input (not constant)
__global__ void acc_from_input(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = in[0];   // init from runtime value
        for (int i = 1; i < n; i++) {
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Nested loops both with trip count <= 16
__global__ void nested_small_loops(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                total += r * 3 + c;   // 0+1+2+3+4+5+6+7+8 = 36
            }
        }
        out[0] = total;
    }
}

// Loop with multiple loop-carried variables
__global__ void multi_carry(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 1;
        // Fibonacci-like: a,b → b,a+b for 8 iterations
        for (int i = 0; i < 8; i++) {
            int tmp = a + b;
            a = b;
            b = tmp;
        }
        out[0] = a;   // fib(8) = 21
        out[1] = b;   // fib(9) = 34
    }
}
