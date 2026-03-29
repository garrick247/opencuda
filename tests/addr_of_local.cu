// Regression: &scalar_local and &arr[idx] address-of operator
// Without fix: &local_val for scalar locals returned the value (not address),
//   corrupting the pointer variable type to INT32 → *ptr += 1 then failed as
//   ParseError "unexpected token '+='".
// Fix: &scalar_local spills the variable to .local memory and returns a pointer
//   to that slot; &ptr[idx] already worked, &ptr_var returns ptr itself.

__device__ void increment(int *p) { *p += 1; }
__device__ void set_pair(float *lo, float *hi, float a, float b) {
    *lo = a < b ? a : b;
    *hi = a > b ? a : b;
}

// &scalar_local: address of a local scalar variable
__global__ void addr_local_scalar(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int local_val = in[tid];
        int *ptr = &local_val;
        *ptr += 1;
        out[tid] = local_val;
    }
}

// &scalar_local passed to device function
__global__ void addr_local_call(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = tid;
        increment(&x);
        out[tid] = x;
    }
}

// &arr[idx]: address of array element (was already supported, regression guard)
__global__ void addr_arr_elem(int *arr, int n) {
    int tid = threadIdx.x;
    if (tid < n/2) {
        int *a = &arr[tid];
        int *b = &arr[n-1-tid];
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

// Two &scalar_locals as output pointers
__global__ void addr_two_locals(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float lo, hi;
        set_pair(&lo, &hi, a[tid], b[tid]);
        out[tid * 2]     = lo;
        out[tid * 2 + 1] = hi;
    }
}
