// Probe: const pointer qualifiers, __restrict__ on multiple params,
// 10-parameter kernel, local array passed to device fn, and struct
// with multiple field types including pointers.

// ------------------------------------------------------------------
// const T * (read-only data pointer).

__global__ void const_ptr_in(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 3;   // read from const pointer
    }
}

// ------------------------------------------------------------------
// T * const (const pointer itself, but mutable data) — unusual in CUDA.
// Just test parsing; semantics same as T* in practice.

__global__ void ptr_const_data(int * const out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] + 1;
    }
}

// ------------------------------------------------------------------
// const T * const (both pointer and data are const).

__global__ void const_const(int *out, const int * const in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2;
    }
}

// ------------------------------------------------------------------
// __restrict__ on multiple params.

__global__ void multi_restrict(float * __restrict__ out,
                                const float * __restrict__ a,
                                const float * __restrict__ b,
                                int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// 10-parameter kernel.

__global__ void ten_params(float *out,
                            float *a, float *b, float *c, float *d,
                            float *e, float *f, float *g, float *h,
                            int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid] + c[tid] + d[tid]
                 + e[tid] + f[tid] + g[tid] + h[tid];
    }
}

// ------------------------------------------------------------------
// Local array passed to device function expecting pointer.

__device__ int sum_array(int *arr, int len) {
    int s = 0;
    for (int i = 0; i < len; i++) s += arr[i];
    return s;
}

__global__ void local_arr_to_fn(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int local[4] = {tid, tid+1, tid+2, tid+3};
        out[tid] = sum_array(local, 4);   // 4*tid + 6
    }
}

// ------------------------------------------------------------------
// Struct with mixed field types including pointer.

struct Buffer {
    float *data;
    int    len;
    int    stride;
};

__global__ void struct_with_ptr(int *out, float *mem, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Buffer buf;
        buf.data   = mem + tid * 4;
        buf.len    = 4;
        buf.stride = 1;
        float s = 0.0f;
        for (int i = 0; i < buf.len; i++) {
            s += buf.data[i * buf.stride];
        }
        out[tid] = (int)s;
    }
}

// ------------------------------------------------------------------
// Struct passed by value to device fn.

struct Range { int lo; int hi; };

__device__ int range_size(struct Range r) {
    return r.hi - r.lo;
}

__global__ void struct_by_value_call(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Range r;
        r.lo = tid;
        r.hi = tid + 10;
        out[tid] = range_size(r);   // 10
    }
}

// ------------------------------------------------------------------
// Function with unsigned int return type.

__device__ unsigned int compute_mask(unsigned int v) {
    return v | (v << 16);
}

__global__ void uint_return_fn(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = compute_mask(in[tid]);
    }
}

// ------------------------------------------------------------------
// Complex default parameter pattern: multiple const qualifiers.

__global__ void const_int_param(int *out, int n, const int scale,
                                  const int offset) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = tid * scale + offset;
    }
}
