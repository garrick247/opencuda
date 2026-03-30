// Probe: edge cases in the new recursive device func support —
// mutual recursion (A calls B, B calls A), recursive with pointer param,
// recursive with multiple return paths, recursive with loop inside,
// non-recursive function that's detected as on-cycle by false positive.

// ------------------------------------------------------------------
// Mutual recursion: is_even calls is_odd, is_odd calls is_even.

__device__ int is_even(int n);
__device__ int is_odd(int n);

__device__ int is_even(int n) {
    if (n == 0) return 1;
    return is_odd(n - 1);
}

__device__ int is_odd(int n) {
    if (n == 0) return 0;
    return is_even(n - 1);
}

__global__ void mutual_recursion_kernel(int *out_even, int *out_odd, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_even[tid] = is_even(in[tid]);
        out_odd[tid]  = is_odd(in[tid]);
    }
}

// ------------------------------------------------------------------
// Recursive function with multiple return paths.

__device__ int gcd(int a, int b) {
    if (b == 0) return a;
    if (a == 0) return b;
    if (a == b) return a;
    if (a > b) return gcd(a - b, b);
    return gcd(a, b - a);
}

__global__ void gcd_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = gcd(a[tid], b[tid]);
}

// ------------------------------------------------------------------
// Recursive with pointer output param (accumulator).

__device__ void sum_recursive(int *result, int *arr, int len) {
    if (len <= 0) return;
    *result += arr[len - 1];
    sum_recursive(result, arr, len - 1);
}

__global__ void sum_recursive_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int result = 0;
        int buf[8];
        for (int k = 0; k < 8; k++) buf[k] = in[(tid + k) % n];
        sum_recursive(&result, buf, 8);
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Recursive with loop inside: recursive merge sort step.

__device__ int recursive_sum_with_loop(int *arr, int start, int end) {
    if (end - start <= 1) return (start < end) ? arr[start] : 0;
    int mid = (start + end) / 2;
    int left = recursive_sum_with_loop(arr, start, mid);
    int right = recursive_sum_with_loop(arr, mid, end);
    return left + right;
}

__global__ void rec_sum_loop_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[8];
        for (int k = 0; k < 8; k++) buf[k] = in[(tid + k) % n];
        out[tid] = recursive_sum_with_loop(buf, 0, 8);
    }
}
