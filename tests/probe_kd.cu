// Probe: device function with internal for-loop (inlining + inner scope),
// two kernels sharing one __device__ function (independent inline each),
// variable bit-shift amounts,
// device function returning result of its own for-loop

// Device function with an internal loop
__device__ int sum_range(int *arr, int lo, int hi) {
    int s = 0;
    for (int i = lo; i < hi; i++) {
        s += arr[i];
    }
    return s;
}

// Kernel 1: calls sum_range once
__global__ void kernel_sum_range_a(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        *out = sum_range(in, 0, n);
    }
}

// Kernel 2: calls sum_range twice with different ranges
// Tests that each inline is independent
__global__ void kernel_sum_range_b(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mid = n / 2;
        int lo_sum = sum_range(in, 0, mid);
        int hi_sum = sum_range(in, mid, n);
        out[0] = lo_sum;
        out[1] = hi_sum;
        out[2] = lo_sum + hi_sum;
    }
}

// Variable bit-shift: shift amount comes from array
__global__ void variable_shift(unsigned int *out, unsigned int *in,
                                unsigned int *shifts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int k = shifts[tid] & 31;  // clamp to valid range
        out[tid] = v << k;
    }
}

// Device function using local variable with same name as caller's variable
__device__ int pop_count_byte(int b) {
    int count = 0;                  // 'count' — same name caller might use
    for (int i = 0; i < 8; i++) {  // 'i' — caller also uses 'i'
        count += (b >> i) & 1;
    }
    return count;
}

__global__ void pop_count_array(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;              // caller also has 'count'
        for (int i = 0; i < n; i++) {  // caller has 'i'
            count += pop_count_byte(in[i] & 0xFF);
        }
        *out = count;
    }
}
