// Test: recursive __device__ functions compiled as PTX .func

// Direct recursion: factorial
__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

__global__ void fact_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = factorial(in[tid]);
}

// Double self-recursion: fibonacci
__device__ int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void fib_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = fibonacci(in[tid]);
}

// Non-recursive helper calling recursive function
__device__ int safe_factorial(int n) {
    if (n < 0) return -1;
    if (n > 12) return -1;  // overflow guard
    return factorial(n);
}

__global__ void safe_fact_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = safe_factorial(in[tid]);
}
