// Probe: pointer arithmetic correctness,
// negative index sign extension (INT32 vs UINT32),
// pointer differences and comparisons,
// pointer passed to device function and modified

// Stencil with negative offsets — tests signed index widening
__global__ void stencil3(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid > 0 && tid < n - 1) {
        out[tid] = 0.333f * (in[tid - 1] + in[tid] + in[tid + 1]);
    }
}

// Pointer arithmetic: start pointer + offset, read through result
__global__ void offset_read(int *out, int *base, int offset, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *ptr = base + offset;   // pointer + int
        out[tid] = ptr[tid];
    }
}

// Signed vs unsigned index: negative tid should not wrap to huge address
__global__ void signed_index_guard(float *out, float *in, int n) {
    int tid = (int)threadIdx.x - 16;  // can be negative
    if (tid >= 0 && tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

// Two-pointer distance computation (element count between pointers)
// Simulated with integer arithmetic since we don't have pointer subtraction
__global__ void array_range_sum(int *out, int *in, int start, int end, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = start; i < end && i < n; i++) {
            sum += in[i];
        }
        *out = sum;
    }
}
