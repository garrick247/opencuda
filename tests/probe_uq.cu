// Probe: pointer arithmetic edge cases, negative indexing, multi-dim
// global arrays, __ldg cache hints, and size_t arithmetic.

// ------------------------------------------------------------------
// Negative pointer offset (ptr[-1], ptr[-2] style).

__global__ void negative_index(float *out, float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0 && tid < n) {
        // Access previous element via negative offset
        float prev = in[tid - 1];
        float curr = in[tid];
        out[tid] = curr - prev;  // forward difference
    }
    if (tid == 0) {
        out[0] = 0.0f;
    }
}

// ------------------------------------------------------------------
// Pointer arithmetic with stride > 1.

__global__ void stride_access(float *out, float *in, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * stride;
    if (base + stride - 1 < n) {
        float acc = 0.0f;
        for (int i = 0; i < stride; i++) {
            acc += in[base + i];
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// __ldg read-only cache hint.

__global__ void ldg_access(float *out, const float * __restrict__ in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = __ldg(&in[tid]);
        out[tid] = v * 2.0f + 1.0f;
    }
}

// ------------------------------------------------------------------
// __ldg on integer array (int read-only hint).

__global__ void ldg_int(int *out, const int * __restrict__ idx,
                          const float * __restrict__ vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = __ldg(&idx[tid]);
        float v = __ldg(&vals[i]);
        out[tid] = (int)(v * 10.0f);
    }
}

// ------------------------------------------------------------------
// __ldg on struct array field.

struct Packed {
    float val;
    int flag;
};

__global__ void ldg_struct(float *out, const Packed * __restrict__ data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = __ldg(&data[tid].val);
        int f   = __ldg(&data[tid].flag);
        out[tid] = f ? v * 2.0f : -v;
    }
}

// ------------------------------------------------------------------
// size_t arithmetic for large array indexing.

__global__ void size_t_index(float *out, float *in, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        size_t mirror = n - 1 - tid;
        out[tid] = in[mirror];
    }
}

// ------------------------------------------------------------------
// ptrdiff_t style (difference of two pointers → signed 64-bit).

__global__ void ptr_diff(long long *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Pointer difference gives element count
        // We compute the byte distance and divide by sizeof(float)
        long long diff = (long long)((unsigned long long)(&b[tid]) -
                                      (unsigned long long)(&a[tid]));
        out[tid] = diff / 4LL;  // sizeof(float) = 4
    }
}

// ------------------------------------------------------------------
// Mixed address space pointer coerce (global to local copy).

__global__ void global_to_local_copy(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Copy chunk from global to registers, then process
        float buf[4];
        int base = (tid / 4) * 4;
        for (int i = 0; i < 4; i++) {
            if (base + i < n)
                buf[i] = in[base + i];
            else
                buf[i] = 0.0f;
        }
        float acc = 0.0f;
        for (int i = 0; i < 4; i++) {
            acc += buf[i];
        }
        out[tid] = acc;
    }
}
