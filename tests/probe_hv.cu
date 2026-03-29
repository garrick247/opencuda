// Probe: cooperative groups patterns — grid.sync() style,
// __nanosleep, clock64(), clock() intrinsics,
// __trap() and __brkpt()

__global__ void clock_measure(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long t0 = clock64();
        // Do some work
        int sum = 0;
        for (int i = 0; i < 100; i++) sum += i;
        long long t1 = clock64();
        out[tid] = t1 - t0 + sum;
    }
}

// __nanosleep (Turing+)
__global__ void sleep_kernel(int *out, int *in, int ns, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __nanosleep(ns);
        out[tid] = in[tid];
    }
}

// volatile load via intrinsic pattern
__global__ void volatile_load(int *out, volatile int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid];
    }
}
