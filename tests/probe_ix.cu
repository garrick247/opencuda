// Probe: complex loop-carried expressions,
// variables used as both loop counter AND accumulator,
// loop exit condition depends on a variable modified in both branches,
// Fibonacci-style dual accumulators

// Counter doubles as accumulator: total += i every iteration
__global__ void counter_as_accum(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += i;
        }
        out[0] = sum;
        out[1] = n;  // n unchanged
    }
}

// Fibonacci: a,b both updated each iteration
__global__ void fib_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        if (n <= 0) { out[0] = 0; return; }
        if (n == 1) { out[0] = 1; return; }
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            int tmp = a + b;
            a = b;
            b = tmp;
        }
        out[0] = b;
    }
}

// Dual pointers narrowing: both lo and hi updated each iteration
__global__ void two_sum_loop(int *result, int *arr, int target, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int s = arr[lo] + arr[hi];
            if (s == target) {
                result[0] = lo;
                result[1] = hi;
                return;
            }
            if (s < target) lo++;
            else hi--;
        }
        result[0] = -1;
        result[1] = -1;
    }
}

// Loop with quadratic stride: i grows non-linearly
__global__ void quadratic_stride(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int step = 1;
        for (int i = 0; i < n; i += step) {
            sum += i;
            step++;  // step increases every iteration
        }
        *out = sum;
    }
}
