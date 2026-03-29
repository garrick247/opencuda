// Probe: patterns involving address-of (&) in non-atomic contexts,
// multiple pointer levels, and pointer comparison patterns.

// ------------------------------------------------------------------
// Take address of local, pass to device function writing via pointer.

__device__ void fill3(int *p, int a, int b, int c) {
    p[0] = a; p[1] = b; p[2] = c;
}

__global__ void local_addr_fill(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[3];
        fill3(buf, in[tid], in[tid]+1, in[tid]+2);
        out[tid * 3 + 0] = buf[0];
        out[tid * 3 + 1] = buf[1];
        out[tid * 3 + 2] = buf[2];
    }
}

// ------------------------------------------------------------------
// Pointer to local scalar (out-param pattern).

__device__ void compute_pair(int v, int *lo, int *hi) {
    *lo = v < 0 ? v : 0;
    *hi = v > 0 ? v : 0;
}

__global__ void ptr_outparam(int *lout, int *hout, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lo, hi;
        compute_pair(in[tid], &lo, &hi);
        lout[tid] = lo;
        hout[tid] = hi;
    }
}

// ------------------------------------------------------------------
// Pointer arithmetic: advance pointer and dereference.

__global__ void ptr_advance(int *out, int *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = in + tid * stride;
        int sum = 0;
        for (int i = 0; i < stride; i++) {
            sum += p[i];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Pointer swap (exchange two pointers' targets).

__device__ void ptr_swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

__global__ void sort2(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid * 2 + 0];
        int b = in[tid * 2 + 1];
        if (a > b) ptr_swap(&a, &b);
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}

// ------------------------------------------------------------------
// Pointer-indexed reduction: sum of strided elements.

__global__ void stride_reduce(int *out, int *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = tid;
        int acc = 0;
        while (base < n * k) {
            acc += in[base];
            base += n;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Pointer diff: used for bounds check.

__global__ void ptr_diff_check(int *out, int *start, int *end_ptr, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Number of elements between start and end_ptr
        long long count = (long long)(end_ptr - start);
        out[tid] = (tid < (int)count) ? start[tid] : -1;
    }
}

// ------------------------------------------------------------------
// Array-of-pointers pattern (pointer arrays on local stack).

__global__ void arr_of_ptrs(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *ptrs[3];
        ptrs[0] = a + tid;
        ptrs[1] = b + tid;
        ptrs[2] = c + tid;
        float sum = 0.0f;
        for (int i = 0; i < 3; i++) {
            sum += *ptrs[i];
        }
        out[tid] = sum;
    }
}
