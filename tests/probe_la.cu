// Probe: multi-return device function via output pointer,
// device function that modifies multiple values,
// device function with loop inside (no early return),
// chained device function calls where result feeds next call's pointer

// Device function using output pointer (workaround for multi-return)
__device__ void minmax(int *a, int *b) {
    if (*a > *b) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

__global__ void sort2(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        int i = tid * 2;
        int x = in[i];
        int y = in[i + 1];
        minmax(&x, &y);
        out[i]     = x;
        out[i + 1] = y;
    }
}

// Device function with internal loop
__device__ int dot_product(int *a, int *b, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

__global__ void dot_kernel(int *out, int *a, int *b, int n, int len) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dot_product(a + tid * len, b + tid * len, len);
    }
}

// Device function that conditionally updates output pointer
__device__ int clamp_sum(int *arr, int n, int lo, int hi) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        int v = arr[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        s += v;
    }
    return s;
}

__global__ void clamped_reduce(int *out, int *in, int n, int lo, int hi) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = clamp_sum(in, n, lo, hi);
    }
}

// Two device functions, first result used as array index for second call
__device__ int find_max_idx(int *arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] > arr[best]) best = i;
    }
    return best;
}

__device__ int val_at(int *arr, int idx) {
    return arr[idx];
}

__global__ void max_value(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int idx = find_max_idx(in, n);
        out[0] = val_at(in, idx);
        out[1] = idx;
    }
}
