// Probe: __ldg on integer types, __float_as_int/__int_as_float bit-cast,
// atomicInc/atomicDec, printf with pointer, and saturating arithmetic.

// ------------------------------------------------------------------
// __ldg on int/uint/long long (non-coherent load via texture cache).

__global__ void ldg_int(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]) + 1;
    }
}

__global__ void ldg_uint(unsigned int *out, const unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]) * 2u;
    }
}

__global__ void ldg_ll(long long *out, const long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&in[tid]) + 1LL;
    }
}

// ------------------------------------------------------------------
// __float_as_int / __int_as_float: bit-cast via PTX mov.

__global__ void float_as_int_kernel(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __float_as_int(in[tid]);
    }
}

__global__ void int_as_float_kernel(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __int_as_float(in[tid]);
    }
}

// ------------------------------------------------------------------
// __double_as_longlong / __longlong_as_double: 64-bit bit-cast.

__global__ void double_as_ll(long long *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __double_as_longlong(in[tid]);
    }
}

__global__ void ll_as_double(double *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __longlong_as_double(in[tid]);
    }
}

// ------------------------------------------------------------------
// atomicInc / atomicDec (wrapping increment/decrement).

__global__ void atomic_inc_dec(unsigned int *counters, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // atomicInc wraps at modulus; atomicDec wraps at 0
        atomicInc(&counters[0], 100u);
        atomicDec(&counters[1], 100u);
    }
}

// ------------------------------------------------------------------
// Saturating adds: use PTX add.sat (via __sad / clamp pattern).

__global__ void clamp_add(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = a[tid] + b[tid];
        // Clamp to [0, 255]
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// __sad: sum of absolute differences.

__global__ void sad_kernel(unsigned int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // __sad(x, y, z) = |x-y| + z
        out[tid] = __sad(a[tid], b[tid], 0u);
    }
}

// ------------------------------------------------------------------
// printf with pointer argument (%p).

__global__ void printf_ptr(int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("ptr=%p n=%d\n", (void*)data, n);
    }
}
