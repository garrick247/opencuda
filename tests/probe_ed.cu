// Probe: Patterns that test the for-loop with empty body
// - for (init; cond; inc) ; (empty body)
// - while (cond) ; (empty body)  
// - for loop where body is a single compound statement with just semicolons

__global__ void empty_loop_body(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Find count of steps to reduce tid to 0 via Collatz-like rules
        int v = tid + 1;
        int steps = 0;
        while (v != 1 && steps < 100) {
            if (v % 2 == 0) v /= 2;
            else v = v * 3 + 1;
            steps++;
        }
        out[tid] = steps;
    }
}

// Loop with just a semicolon body (count iterations)
__global__ void count_down_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = n;
        int count = 0;
        // for (; i > 0; i--, count++); — semicolon body with comma in inc
        for (; i > 0; i--, count++)
            ;  // empty body via semicolon
        out[tid] = count;
    }
}
