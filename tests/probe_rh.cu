// Probe: printf with various arg types, __syncwarp, __activemask,
// type-punning via union, and texture-less read patterns.

// ------------------------------------------------------------------
// printf with various format arg types: int, float, long long, double.

__global__ void printf_types(int *data, float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   iv = data[tid];
        float fv = fdata[tid];
        if (iv > 100) {
            printf("tid=%d int=%d float=%f\n", tid, iv, fv);
        }
    }
}

// ------------------------------------------------------------------
// printf with long long.

__global__ void printf_ll(long long *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = data[tid];
        if (v > 1000000LL) {
            printf("tid=%d ll=%lld\n", tid, v);
        }
    }
}

// ------------------------------------------------------------------
// __syncwarp(): synchronize within a warp.

__global__ void syncwarp_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid] * 2.0f;
        __syncwarp();
        out[tid] = v + 1.0f;
    }
}

// ------------------------------------------------------------------
// __activemask(): get mask of active threads in warp.

__global__ void activemask_kernel(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int mask = __activemask();
        out[tid] = mask;
    }
}

// ------------------------------------------------------------------
// Type-punning via union: float ↔ int bit reinterpretation.

union FloatInt {
    float f;
    int   i;
};

__global__ void float_to_bits(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        FloatInt fi;
        fi.f = in[tid];
        out[tid] = fi.i;
    }
}

__global__ void bits_to_float(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        FloatInt fi;
        fi.i = in[tid];
        out[tid] = fi.f;
    }
}

// ------------------------------------------------------------------
// Multiple __syncthreads in same kernel.

__global__ void multi_sync(float *out, float *in, int n) {
    __shared__ float s1[32];
    __shared__ float s2[32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    // Phase 1: write to s1
    if (tid < n && lane < 32) s1[lane] = in[tid] * 2.0f;
    __syncthreads();
    // Phase 2: read s1, write to s2
    if (tid < n && lane < 32) s2[lane] = s1[(lane + 1) % 32] + 1.0f;
    __syncthreads();
    // Phase 3: read s2, write output
    if (tid < n && lane < 32) out[tid] = s2[(lane + 2) % 32];
}

// ------------------------------------------------------------------
// __trap() in unreachable else clause.

__global__ void trap_unreachable(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid] % 3;
        int r;
        if      (v == 0) r = 0;
        else if (v == 1) r = 1;
        else             r = 2;
        out[tid] = r;
    }
}
