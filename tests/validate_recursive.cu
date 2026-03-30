// Recursive device function runtime validation.
// Tests that PTX .func/.call produces correct results.

__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Wrapper to match (int *out, int *a, int *b, int n) signature.
__global__ void factorial_k(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // Clamp input to [0, 12] to avoid overflow
        int v = a[gid] % 13;
        if (v < 0) v = -v;
        out[gid] = factorial(v);
    }
}

__device__ int fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void fibonacci_k(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = a[gid] % 15;
        if (v < 0) v = -v;
        out[gid] = fibonacci(v);
    }
}

__device__ int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

__global__ void gcd_k(int *out, int *a, int *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int va = a[gid]; if (va < 0) va = -va; if (va == 0) va = 1;
        int vb = b[gid]; if (vb < 0) vb = -vb; if (vb == 0) vb = 1;
        out[gid] = gcd(va, vb);
    }
}
