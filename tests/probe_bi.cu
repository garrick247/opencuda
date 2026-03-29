// Probe: More unusual C syntax patterns
// - Ternary operator with side effects as statement
// - Conditional increment: if (cond) x++;
// - Assignment as condition: while ((v = *ptr++) != 0)
// - Unary address-of on array element: &arr[i]

__global__ void cond_increment(int *out, int *flags, int *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = vals[tid];
        if (flags[tid]) v++;
        out[tid] = v;
    }
}

// &arr[i] — address of array element as function arg
__device__ void scale_inplace(float *p, float s) {
    *p = *p * s;
}

__global__ void addr_of_array_elem(float *arr, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        scale_inplace(&arr[tid], scale);
    }
}

// Nested ternary as expression statement (result discarded, but side effect in args)
__global__ void ternary_side_effect(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        // Assign from ternary
        int r = (x > y) ? (x - y) : (y - x);
        out[tid] = r;
    }
}
