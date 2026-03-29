// Probe: for-loop with complex condition and multiple loop variables
// - Loop var used in both condition and body as array index
// - Loop where condition is a function call result
// - Nested loops with same variable name (shadowing)

__device__ int count_bits(unsigned int v) {
    int count = 0;
    while (v) {
        count += (int)(v & 1u);
        v >>= 1;
    }
    return count;
}

__global__ void popcount_loop(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = count_bits(in[tid]);
    }
}

// Loop condition that's a comparison between two loop vars
__global__ void two_var_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lo = 0, hi = n - 1;
        int steps = 0;
        while (lo < hi) {
            lo++;
            hi--;
            steps++;
        }
        out[tid] = steps;
    }
}

// Fibonacci via loop (classic correctness test)
__global__ void fibonacci(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        int a = 0, b = 1;
        for (int i = 0; i < n; i++) {
            int tmp = a + b;
            a = b;
            b = tmp;
        }
        out[0] = a;
    }
}
