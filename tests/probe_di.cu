// Probe: For loop where condition is a compound boolean with function calls
// and patterns where a device function returns true/false used as loop condition

__device__ int in_range(int v, int lo, int hi) {
    return v >= lo && v < hi;
}

__device__ float safe_recip(float x) {
    return (x != 0.0f) ? (1.0f / x) : 0.0f;
}

__global__ void filter_range(int *out, int *in, int lo, int hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = in_range(v, lo, hi) ? v : -1;
    }
}

// Function call in loop condition
__global__ void loop_func_cond(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        int i = tid;
        while (in_range(i, 0, n)) {
            sum += safe_recip(in[i]);
            i++;
            if (i - tid >= 8) break;
        }
        out[tid] = sum;
    }
}

// Ternary in array index with function call
__global__ void ternary_index_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = in_range(tid * 2, 0, n) ? tid * 2 : tid;
        out[tid] = in[idx];
    }
}
